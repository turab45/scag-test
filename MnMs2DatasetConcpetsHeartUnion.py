# LA + SAX

# concepts for both LV and RV

import os
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional
from scag_dataset_utils import dataset_paths, resolve_phase_matched_gt, stable_directory_entries

MNM2_ROOT = '/media/kislay/New Volume/Turab/data/MnM2/'


# ─────────────────────────────────────────────────────────────────────────────
# GT label constants (MnMs-2 convention)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_BG  = 0
LABEL_LV  = 1
LABEL_MYO = 2
LABEL_RV  = 3


def _heart_union_bbox(gt_slice: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return the tight bounding box around the non-background heart union."""
    heart_mask = gt_slice > LABEL_BG
    if not np.any(heart_mask):
        return None

    rows, cols = np.where(heart_mask)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _crop_to_bbox(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a 2D array to a (row0, row1, col0, col1) bounding box."""
    row0, row1, col0, col1 = bbox
    return array[row0:row1, col0:col1]


def _resize_mask(mask: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize a discrete mask with nearest-neighbor interpolation."""
    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST)).astype(mask.dtype)


def compute_lax_concepts(gt_slice: np.ndarray) -> Optional[dict[str, np.ndarray]]:
    """
    Rotation-robust LAX concept derivation.

    Derives:
        LV_basal,  LV_mid,  LV_apical
        MYO_basal, MYO_mid, MYO_apical
        RV_basal,  RV_mid,  RV_apical

    Key fix:
    - estimate the cardiac long axis from LV+MYO using PCA
    - split along that axis instead of raw rows/cols
    - orient the axis so the RV side is treated as basal
    """
    heart_mask = gt_slice > LABEL_BG
    if not np.any(heart_mask):
        return None

    lv_mask  = (gt_slice == LABEL_LV)
    myo_mask = (gt_slice == LABEL_MYO)
    rv_mask  = (gt_slice == LABEL_RV)

    # Use LV+MYO as the reference cardiac structure for long-axis estimation
    ref_mask = lv_mask | myo_mask
    if not np.any(ref_mask):
        ref_mask = heart_mask

    coords_ref = np.argwhere(ref_mask)   # (row, col)
    center = coords_ref.mean(axis=0)

    # PCA / principal axis
    X = coords_ref - center
    if len(coords_ref) < 2:
        return None

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]   # unit vector in (row, col)

    # Orient axis so that the RV side is "basal"
    # If RV is absent, keep default orientation.
    # if np.any(rv_mask):
    #     rv_center = np.argwhere(rv_mask).mean(axis=0)
    #     if np.dot(rv_center - center, axis) < 0:
    #         axis = -axis
    # Orient axis using LV shape:
    # basal = wider LV end, apical = narrower LV end
    coords_lv = np.argwhere(lv_mask) if np.any(lv_mask) else coords_ref

    # orthogonal axis
    orth = np.array([-axis[1], axis[0]])

    proj_lv = (coords_lv - center) @ axis
    orth_lv = (coords_lv - center) @ orth

    lo_lv, hi_lv = proj_lv.min(), proj_lv.max()
    span_lv = hi_lv - lo_lv

    if span_lv > 1e-6:
        # take small end-windows near both extremes of the LV
        frac = 0.15
        low_end  = proj_lv <= (lo_lv + frac * span_lv)
        high_end = proj_lv >= (hi_lv - frac * span_lv)

        # width at each end measured along orthogonal direction
        def end_width(mask):
            vals = orth_lv[mask]
            if vals.size < 2:
                return 0.0
            return vals.max() - vals.min()

        width_low = end_width(low_end)
        width_high = end_width(high_end)

        # We want: low projection = apical, high projection = basal
        # If low end is actually wider, flip axis.
        if width_low > width_high:
            axis = -axis

    # Project reference structure onto the long axis
    proj_ref = (coords_ref - center) @ axis
    lo, hi = proj_ref.min(), proj_ref.max()

    if hi - lo < 1e-6:
        return None

    t1 = lo + (hi - lo) / 3.0
    t2 = lo + 2.0 * (hi - lo) / 3.0

    # Build shared longitudinal bands by projecting every heart pixel
    H, W = gt_slice.shape
    basal_band  = np.zeros((H, W), dtype=bool)
    mid_band    = np.zeros((H, W), dtype=bool)
    apical_band = np.zeros((H, W), dtype=bool)

    coords_all = np.argwhere(heart_mask)
    proj_all = (coords_all - center) @ axis

    # Since axis is oriented toward RV/basal:
    # low projection  -> apical
    # mid projection  -> mid
    # high projection -> basal
    apical_coords = coords_all[proj_all < t1]
    mid_coords    = coords_all[(proj_all >= t1) & (proj_all < t2)]
    basal_coords  = coords_all[proj_all >= t2]

    apical_band[apical_coords[:, 0], apical_coords[:, 1]] = True
    mid_band[mid_coords[:, 0], mid_coords[:, 1]] = True
    basal_band[basal_coords[:, 0], basal_coords[:, 1]] = True

    return {
        'LV_basal':   lv_mask  & basal_band,
        'LV_mid':     lv_mask  & mid_band,
        'LV_apical':  lv_mask  & apical_band,

        'MYO_basal':  myo_mask & basal_band,
        'MYO_mid':    myo_mask & mid_band,
        'MYO_apical': myo_mask & apical_band,

        'RV_basal':   rv_mask  & basal_band,
        'RV_mid':     rv_mask  & mid_band,
        'RV_apical':  rv_mask  & apical_band,
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component from a boolean mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    labels, n_comp = ndi.label(mask)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labels == keep


def _keep_top_components(mask: np.ndarray, k: int = 2, min_size: int = 25) -> np.ndarray:
    """Keep up to top-k largest connected components above min_size."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    labels, n_comp = ndi.label(mask)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    order = np.argsort(counts)[::-1]

    out = np.zeros_like(mask, dtype=bool)
    kept = 0
    for lab in order:
        if lab == 0:
            continue
        if counts[lab] < min_size:
            continue
        out |= (labels == lab)
        kept += 1
        if kept >= k:
            break

    return out



# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MnMsDatasetLAX(Dataset):
    """
    Dataset loader for MnM2 cardiac MRI data.
    Loads only 2D slices that have a corresponding non-empty ground-truth mask.

    __getitem__ returns a 9-tuple:
        img_tensor    – (3, 224, 224) float32 tensor
        gt_slice_orig – (H, W) int ndarray, original-resolution GT mask
        label_idx     – int, disease class index
        disease_label – str
        patient_folder– str
        slice_idx     – int
        file_name     – str
        view          – 'SA' | 'LA'
        concepts      – dict of concept masks, or None for SA.
                        For crop_to_heart_union=True:
                        - derive_lax_concepts=False -> {'LV', 'MYO', 'RV'}
                        - derive_lax_concepts=True  -> 9 derived LAX masks
    """

    def __init__(self, data_dir, transform=None, target_shape=(3, 224, 224),
                 disease_filter=None, max_images=None, include_heart_union=False, include_heart_union_only=False,
                 crop_to_heart_union=False, derive_lax_concepts=False, include_sa=False, include_both_sa_la=False):
        self.data_dir, self.labels_csv = dataset_paths(data_dir)
        self.transform = transform
        self.target_shape = target_shape
        self.image_slices = []
        self.labels = self.load_labels()
        self.label_mapping = self.create_label_mapping()
        self.skipped_missing_gt = 0
        self.skipped_empty_masks = 0
        self.include_heart_union = include_heart_union
        self.include_heart_union_only = include_heart_union_only
        self.crop_to_heart_union = crop_to_heart_union
        self.derive_lax_concepts = derive_lax_concepts
        self.include_sa = include_sa
        self.include_both_sa_la = include_both_sa_la

        for patient_folder in stable_directory_entries(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            if disease_filter is not None:
                patient_label = self.labels.get(int(patient_folder))
                if patient_label != disease_filter:
                    continue

            for file in stable_directory_entries(patient_path):
                if file.endswith(".nii.gz") and "gt" not in file and "CINE" not in file:
                    nifti_path = os.path.join(patient_path, file)
                    gt_path = self.get_gt_path(patient_folder, file)
                    if gt_path is None:
                        self.skipped_missing_gt += 1
                        continue

                    img = nib.load(nifti_path)
                    gt_img = nib.load(gt_path)
                    num_slices = min(img.shape[2], gt_img.shape[2])
                    gt_data = gt_img.get_fdata()

                    for slice_idx in range(num_slices):
                        if not self.slice_has_mask(gt_data, slice_idx):
                            self.skipped_empty_masks += 1
                            continue
                        self.image_slices.append(
                            (nifti_path, gt_path, slice_idx, patient_folder, file)
                        )
                        if max_images is not None and len(self.image_slices) >= max_images:
                            break

                    if max_images is not None and len(self.image_slices) >= max_images:
                        break

            if max_images is not None and len(self.image_slices) >= max_images:
                break

    def __len__(self):
        return len(self.image_slices)

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_gt_path(self, patient_folder, file_name):
        return resolve_phase_matched_gt(self.data_dir, patient_folder, file_name)

    def slice_has_mask(self, gt_data, slice_idx):
        if slice_idx >= gt_data.shape[2]:
            return False
        return np.any(gt_data[:, :, slice_idx] > 0)

    def load_labels(self):
        df = pd.read_csv(self.labels_csv)
        return {
            int(row['SUBJECT_CODE']): row['DISEASE']
            for _, row in df.iterrows()
            if pd.notna(row['SUBJECT_CODE'])
        }

    def create_label_mapping(self):
        unique_labels = sorted(set(self.labels.values()))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Label Mapping: {label_mapping}")
        return label_mapping

    def preprocess_slice(self, slice_data):
        slice_data = np.squeeze(slice_data)
        if len(slice_data.shape) != 2:
            raise ValueError(f"Expected 2D slice, got shape {slice_data.shape}")
        slice_data = np.nan_to_num(slice_data, nan=0.0).astype(np.float32)
        mn, mx = slice_data.min(), slice_data.max()
        slice_data = np.zeros_like(slice_data) if (mx - mn) < 1e-5 \
                     else (slice_data - mn) / (mx - mn)
        pil = Image.fromarray((slice_data * 255).astype(np.uint8))
        t = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        img_resized = t(pil).repeat(3, 1, 1)
        return img_resized

    def _build_heart_union_concepts(self, gt_slice):
        base_concepts = {
            'LV': gt_slice == LABEL_LV,
            'MYO': gt_slice == LABEL_MYO,
            'RV': gt_slice == LABEL_RV,
        }

        if self.derive_lax_concepts:
            return compute_lax_concepts(gt_slice) or {}

        return base_concepts

    def _resize_concepts(self, concepts, size):
        return {
            name: _resize_mask(mask.astype(np.uint8), size=size).astype(bool)
            for name, mask in concepts.items()
        }

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        nifti_path, gt_path, slice_idx, patient_folder, file_name = self.image_slices[idx]

        # Image tensor
        img = nib.load(nifti_path)
        slice_data = img.get_fdata()[:, :, slice_idx]

        # GT mask at original resolution
        gt_img   = nib.load(gt_path)
        gt_slice = gt_img.get_fdata()[:, :, slice_idx].astype(np.int32)

        concepts_gt_slice = gt_slice

        # View type
        view = 'LA' if '_LA_' in file_name else 'SA'

        # Determine if we should crop for this view
        should_crop = self.crop_to_heart_union
        if self.include_both_sa_la:
            should_crop = True
        elif self.include_sa and view == 'SA':
            should_crop = True

        if should_crop:
            bbox = _heart_union_bbox(gt_slice)
            if bbox is not None:
                slice_data = _crop_to_bbox(slice_data, bbox)
                gt_slice = _crop_to_bbox(gt_slice, bbox)
            concepts_gt_slice = gt_slice

        img_tensor = self.preprocess_slice(slice_data)

        if should_crop:
            target_size = (self.target_shape[2], self.target_shape[1])
            gt_slice = _resize_mask(gt_slice, size=target_size)

        # Concept regions — 3 GT masks for LA (original) or SA/both (new params)
        if should_crop:
            concepts = self._build_heart_union_concepts(concepts_gt_slice)
            concepts = self._resize_concepts(concepts, size=target_size)
        
        else:
            concepts = None

        # Labels
        patient_id    = int(patient_folder)
        disease_label = self.labels[patient_id]
        label_idx     = self.label_mapping[disease_label]

        return img_tensor, gt_slice, label_idx, disease_label, \
               patient_folder, slice_idx, file_name, view, concepts


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

# colour palettes
GT_COLORS    = ['black', '#e63946', "#e87618", "#042d46"]   # BG, LV, MYO, RV
CONCEPT_RGBA = {
    'LV_basal':   (0.95, 0.35, 0.35, 0.55),
    'LV_mid':     (0.85, 0.20, 0.20, 0.55),
    'LV_apical':  (0.65, 0.05, 0.05, 0.55),

    'MYO_basal':  (0.98, 0.78, 0.35, 0.55),
    'MYO_mid':    (0.95, 0.60, 0.20, 0.55),
    'MYO_apical': (0.78, 0.42, 0.05, 0.55),

    'RV_basal':   (0.45, 0.78, 0.98, 0.55),
    'RV_mid':     (0.20, 0.58, 0.95, 0.55),
    'RV_apical':  (0.05, 0.35, 0.78, 0.55),

    'left_lung':  (0.20, 0.85, 0.95, 0.40),
    # 'right_lung': (0.10, 0.65, 0.90, 0.40),
    'chest_wall': (0.95, 0.35, 0.85, 0.35),
    'heart_union': (0.8, 0.6, 0.2, 0.5)
}

def visualize_sample_lax(img_tensor, gt_slice, view, concepts,
                     patient_folder, slice_idx, disease_label, sample_idx):
    """
    Render a 3-panel figure:
      col 1 – raw MRI slice
      col 2 – GT segmentation overlay (LV / MYO / RV)
      col 3 – LAX concept regions (basal / mid / apical), or 'N/A' for SA
    """
    img_np = img_tensor.numpy()[0]          # single channel (H, W)

    n_cols   = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(13, 4.2),
                             facecolor='#1a1a2e')
    fig.suptitle(
        f'[{sample_idx}]  Patient {patient_folder}  •  Slice {slice_idx}  '
        f'•  {disease_label}  •  {view}',
        color='white', fontsize=11, fontweight='bold', y=1.01
    )

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # ── panel 1: raw MRI ─────────────────────────────────────────────────────
    axes[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('MRI slice', color='#ccc', fontsize=9)
    axes[0].axis('off')

    # ── panel 2: GT overlay ──────────────────────────────────────────────────
    axes[1].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    gt_cmap = ListedColormap(GT_COLORS)
    axes[1].imshow(gt_slice, cmap=gt_cmap, vmin=0, vmax=3, alpha=0.55)
    axes[1].set_title('GT segmentation', color='#ccc', fontsize=9)
    axes[1].axis('off')
    legend_handles = [
        mpatches.Patch(color=GT_COLORS[1], label='LV'),
        mpatches.Patch(color=GT_COLORS[2], label='MYO'),
        mpatches.Patch(color=GT_COLORS[3], label='RV'),
    ]
    axes[1].legend(handles=legend_handles, loc='lower right',
                   fontsize=7, framealpha=0.3, labelcolor='white')

    # ── panel 3: concept regions ─────────────────────────────────────────────
        # ── panel 3: concept regions ─────────────────────────────────────────────
    axes[2].imshow(img_np, cmap='gray', vmin=0, vmax=1)

    if concepts is not None:
        present_handles = []

        for name, color in CONCEPT_RGBA.items():
            mask = concepts.get(name, None)
            if mask is None or not np.any(mask):
                continue

            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_224 = np.array(
                mask_pil.resize((224, 224), Image.NEAREST)
            ).astype(bool)

            rgba = np.zeros((224, 224, 4), dtype=np.float32)
            rgba[mask_224] = color
            axes[2].imshow(rgba)

            present_handles.append(
                mpatches.Patch(color=color[:3], label=name.replace('_', ' '))
            )

        axes[2].set_title('LAX concepts (cardiac + contextual)', color='#ccc', fontsize=9)

        if present_handles:
            axes[2].legend(
                handles=present_handles,
                loc='lower right',
                fontsize=6,
                framealpha=0.3,
                labelcolor='white',
                ncol=1
            )
    else:
        axes[2].set_title('Concepts: N/A (SAX slice)', color='#666', fontsize=9)
        axes[2].text(0.5, 0.5, 'SAX view —\nno LAX concepts',
                     ha='center', va='center', color='#555',
                     transform=axes[2].transAxes, fontsize=10)

    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
