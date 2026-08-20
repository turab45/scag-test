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

MNM2_ROOT = '/media/kislay/New Volume/Turab/data/MnM2/'


# ─────────────────────────────────────────────────────────────────────────────
# GT label constants (MnMs-2 convention)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_BG  = 0
LABEL_LV  = 1
LABEL_MYO = 2
LABEL_RV  = 3


def compute_lax_concepts(gt_slice: np.ndarray, concepts:str=None) -> Optional[dict[str, np.ndarray]]:
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

    if concepts == "cardiac":
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
    else:
        concepts = {
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

    return concepts


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


def get_contextual_concepts(slice_data: np.ndarray,
                            gt_slice: np.ndarray,
                            include_background: bool = False,
                            include_heart_union: bool = False,
                            include_heart_union_only: bool = False) -> dict[str, np.ndarray]:
    """
    Derive contextual concepts for LONG-AXIS cardiac MRI slices.

    Returns at least:
      - left_lung
      - right_lung
      - chest_wall

    Optional (set include_background=True):
      - background

    Notes
    -----
    This is a heuristic intensity + morphology approach (not supervised lung/chest
    segmentation). It uses the cardiac GT mask as an exclusion prior.
    """
    img = np.nan_to_num(slice_data.astype(np.float32), nan=0.0)
    H, W = img.shape

    # Normalize image to [0, 1]
    mn, mx = float(img.min()), float(img.max())
    if (mx - mn) < 1e-6:
        img_norm = np.zeros_like(img, dtype=np.float32)
    else:
        img_norm = (img - mn) / (mx - mn)

    empty = np.zeros((H, W), dtype=bool)

    nonzero_vals = img_norm[img_norm > 0]
    if nonzero_vals.size == 0:
        output = {
            'left_lung': empty,
            'right_lung': empty,
            'chest_wall': empty,
        }
        if include_background:
            output['background'] = np.ones((H, W), dtype=bool)
        return output

    s3 = np.ones((3, 3), dtype=bool)
    s5 = np.ones((5, 5), dtype=bool)
    s7 = np.ones((7, 7), dtype=bool)

    # 1) Body mask from low-intensity threshold + cleanup
    fg_thresh = max(0.03, float(np.percentile(nonzero_vals, 12)) * 0.5)
    body_seed = img_norm > fg_thresh
    body_seed = ndi.binary_opening(body_seed, structure=s3)
    body_seed = ndi.binary_closing(body_seed, structure=s7)

    body = _largest_component(body_seed)
    if not np.any(body):
        output = {
            'left_lung': empty,
            'right_lung': empty,
            'chest_wall': empty,
        }
        if include_background:
            output['background'] = np.ones((H, W), dtype=bool)
        return output

    body = ndi.binary_fill_holes(body)
    body = ndi.binary_closing(body, structure=s7)
    body = ndi.binary_fill_holes(body)
    background = ~body

    # 2) Heart exclusion region from GT labels
    heart_union = (gt_slice == LABEL_LV) | (gt_slice == LABEL_MYO) | (gt_slice == LABEL_RV)
    heart_exclusion = ndi.binary_dilation(
        heart_union,
        structure=s3,
        iterations=max(3, int(0.02 * min(H, W)))
    )

    # 3) Build upper-thorax ROI and a central exclusion strip
    body_rows = np.where(np.any(body, axis=1))[0]
    body_cols = np.where(np.any(body, axis=0))[0]
    if body_rows.size == 0 or body_cols.size == 0:
        output = {
            'left_lung': empty,
            'right_lung': empty,
            'chest_wall': empty,
        }
        if include_background:
            output['background'] = background
        if include_heart_union:
            output['heart_union'] = heart_union
        if include_heart_union_only:
            output={'heart_union': heart_union}
        return output

    top_r, bot_r = int(body_rows[0]), int(body_rows[-1])
    left_c, right_c = int(body_cols[0]), int(body_cols[-1])
    body_h = bot_r - top_r + 1
    body_w = right_c - left_c + 1

    upper_limit = min(H - 1, top_r + int(0.80 * body_h))
    upper_body = np.zeros_like(body, dtype=bool)
    upper_body[top_r:upper_limit + 1, left_c:right_c + 1] = body[top_r:upper_limit + 1, left_c:right_c + 1]

    heart_cols = np.where(np.any(heart_union, axis=0))[0]
    if heart_cols.size > 0:
        center_x = int(np.round((heart_cols[0] + heart_cols[-1]) / 2.0))
        heart_w = int(heart_cols[-1] - heart_cols[0] + 1)
    else:
        center_x = int(np.round((left_c + right_c) / 2.0))
        heart_w = max(10, int(0.16 * body_w))

    central_half_w = max(8, int(0.55 * heart_w))
    mediastinal_band = np.zeros_like(body, dtype=bool)
    c0 = max(0, center_x - central_half_w)
    c1 = min(W, center_x + central_half_w + 1)
    mediastinal_band[top_r:upper_limit + 1, c0:c1] = upper_body[top_r:upper_limit + 1, c0:c1]

    # 4) Side-specific search zones and dark-region extraction
    left_zone = np.zeros_like(body, dtype=bool)
    right_zone = np.zeros_like(body, dtype=bool)
    left_zone[top_r:upper_limit + 1, left_c:center_x] = upper_body[top_r:upper_limit + 1, left_c:center_x]
    right_zone[top_r:upper_limit + 1, center_x:right_c + 1] = upper_body[top_r:upper_limit + 1, center_x:right_c + 1]

    left_search = left_zone & (~mediastinal_band) & (~heart_exclusion)
    right_search = right_zone & (~mediastinal_band) & (~heart_exclusion)

    def _extract_single_lung(search_mask: np.ndarray) -> np.ndarray:
        if not np.any(search_mask):
            return np.zeros_like(search_mask, dtype=bool)

        vals = img_norm[search_mask]
        if vals.size == 0:
            return np.zeros_like(search_mask, dtype=bool)

        dark_q = float(np.percentile(vals, 35))
        cand = search_mask & (img_norm <= dark_q)
        cand = ndi.binary_opening(cand, structure=s5)
        cand = ndi.binary_closing(cand, structure=s7)
        cand = ndi.binary_fill_holes(cand)

        return _keep_top_components(cand, k=1, min_size=max(25, int(0.004 * H * W)))

    left_lung = _extract_single_lung(left_search)
    right_lung = _extract_single_lung(right_search)
    lungs_union = left_lung | right_lung

    # 5) Chest wall as peripheral in-body ring excluding lungs and heart
    ring_width = max(4, int(0.03 * min(H, W)))
    eroded = ndi.binary_erosion(body, structure=s3, iterations=ring_width)
    chest_ring = body & (~eroded)
    chest_wall = chest_ring & (~lungs_union) & (~heart_exclusion)
    chest_wall = ndi.binary_opening(chest_wall, structure=s3)
    chest_wall = ndi.binary_closing(chest_wall, structure=s5)

    output = {
        'left_lung': left_lung,
        # 'right_lung': right_lung,
        'chest_wall': chest_wall
    }
    if include_background:
        output['background'] = background
    if include_heart_union:
        output['heart_union'] = heart_union
    if include_heart_union_only:
        output = {'heart_union': heart_union}

    return output

    


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
        concepts      – dict {'basal', 'mid', 'apical'} of bool masks, or None for SA
    """

    def __init__(self, data_dir, transform=None, target_shape=(3, 224, 224),
                 disease_filter=None, max_images=None, include_heart_union=False, include_heart_union_only=False,
                 get_original_concepts=False):
        self.data_dir = data_dir + "dataset/"
        self.labels_csv = data_dir + "dataset_information.csv"
        self.transform = transform
        self.target_shape = target_shape
        self.image_slices = []
        self.labels = self.load_labels()
        self.label_mapping = self.create_label_mapping()
        self.skipped_missing_gt = 0
        self.skipped_empty_masks = 0
        self.include_heart_union = include_heart_union
        self.include_heart_union_only = include_heart_union_only
        self.get_original_concepts = get_original_concepts
        for patient_folder in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            if disease_filter is not None:
                patient_label = self.labels.get(int(patient_folder))
                if patient_label != disease_filter:
                    continue

            for file in os.listdir(patient_path):
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
        patient_dir = os.path.join(self.data_dir, patient_folder)
        view = 'SA' if '_SA_' in file_name else 'LA' if '_LA_' in file_name else None
        if view is None:
            return None
        for phase in ('ED', 'ES'):
            gt_path = os.path.join(patient_dir, f'{patient_folder}_{view}_{phase}_gt.nii.gz')
            if os.path.exists(gt_path):
                return gt_path
        return None

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

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        nifti_path, gt_path, slice_idx, patient_folder, file_name = self.image_slices[idx]

        # Image tensor
        img = nib.load(nifti_path)
        slice_data = img.get_fdata()[:, :, slice_idx]
        img_tensor = self.preprocess_slice(slice_data)

        # GT mask at original resolution
        gt_img   = nib.load(gt_path)
        gt_slice = gt_img.get_fdata()[:, :, slice_idx].astype(np.int32)

        # View type
        view = 'LA' if '_LA_' in file_name else 'SA'

        # Concept regions — only for LAX views
        if view == 'LA' and self.get_original_concepts == False:
            cardiac_concepts = compute_lax_concepts(gt_slice) or {}
            contextual_concepts = get_contextual_concepts(
                slice_data=slice_data,
                gt_slice=gt_slice,
                include_background=False,
                include_heart_union=self.include_heart_union,
                include_heart_union_only=self.include_heart_union_only
            )

            if self.include_heart_union_only:
                # for the cropping thing, we only want the heart union mask as the concept, not the cardiac sub-regions
                concepts = contextual_concepts
            
            concepts = {**cardiac_concepts, **contextual_concepts}
        elif self.get_original_concepts == True:
            concepts = {
                'LV': gt_slice == LABEL_LV,
                'MYO': gt_slice == LABEL_MYO,
                'RV': gt_slice == LABEL_RV
            }
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
