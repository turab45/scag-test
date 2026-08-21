#!/usr/bin/env python3
"""Extract patient-level inputs for independent SCAG evaluation on the GPU machine.

This script uses Turab's uploaded M&Ms-2 loaders and preserves the existing data
and checkpoint paths by default.  Unlike the original scripts, it saves one
activation vector and one channel-by-concept score matrix per sample together
with patient IDs.  That information is required for patient-level cross-fitting.

Examples
--------
# Quick extraction check (3 concepts, cropped, block 1, 8 LA samples)
python extract_scag_patient_data.py --image-mode cropped --concepts 3 --blocks 1 --max-images 8

# Full four-block extraction
python extract_scag_patient_data.py --image-mode cropped --concepts 3 --blocks 1 2 3 4
python extract_scag_patient_data.py --image-mode full    --concepts 3 --blocks 1 2 3 4
python extract_scag_patient_data.py --image-mode cropped --concepts 9 --blocks 1 2 3 4
python extract_scag_patient_data.py --image-mode full    --concepts 9 --blocks 1 2 3 4
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MNM2_ROOT = "/media/kislay/New Volume/Turab/data/MnM2/"
DEFAULT_MODEL_PATH = (
    "/media/kislay/New Volume/Turab/CRAFT/models/best_densenet161_MnMs.pth"
)


def concept_names_for(n_concepts: int) -> list[str]:
    if n_concepts == 3:
        return ["LV", "MYO", "RV"]
    if n_concepts == 9:
        return [
            "LV_basal",
            "LV_mid",
            "LV_apical",
            "MYO_basal",
            "MYO_mid",
            "MYO_apical",
            "RV_basal",
            "RV_mid",
            "RV_apical",
        ]
    raise ValueError("n_concepts must be 3 or 9")


def dataset_spec_for(image_mode: str, n_concepts: int):
    if image_mode == "cropped":
        return (
            "MnMs2DatasetConcpetsHeartUnion",
            {
                "crop_to_heart_union": True,
                "derive_lax_concepts": n_concepts == 9,
            },
        )
    if image_mode == "full":
        return (
            "MnMs2DatasetConcpets",
            {"get_original_concepts": n_concepts == 3},
        )
    raise ValueError("image_mode must be 'cropped' or 'full'")


def validate_payload(payload: dict) -> None:
    required = ["pooled_activations", "concept_scores", "patient_ids", "concept_names"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload missing required keys: {missing}")
    activations = np.asarray(payload["pooled_activations"])
    scores = np.asarray(payload["concept_scores"])
    patients = np.asarray(payload["patient_ids"])
    names = np.asarray(payload["concept_names"])
    if activations.ndim != 2:
        raise ValueError("pooled_activations must be (samples, channels)")
    if scores.ndim != 3:
        raise ValueError("concept_scores must be (samples, channels, concepts)")
    if len(activations) != len(scores) or len(activations) != len(patients):
        raise ValueError(
            "sample dimensions disagree among activations, scores, and patient IDs"
        )
    if activations.shape[1] != scores.shape[1]:
        raise ValueError("channel dimensions disagree")
    if scores.shape[2] != len(names):
        raise ValueError("concept dimension disagrees with concept_names")
    if len(np.unique(patients.astype(str))) < 2:
        raise ValueError("at least two patients are required")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_model(torch, timm, checkpoint_path: str, device: str):
    model = timm.create_model("densenet161", pretrained=False, num_classes=8)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This pipeline requires a PyTorch version supporting torch.load(weights_only=True). "
            "Upgrade PyTorch rather than loading an unsafe pickle checkpoint."
        ) from exc
    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _collate(batch):
    import torch

    imgs, gt_slices, labels, diseases, patients, slices, files, views, concepts = zip(
        *batch
    )
    return (
        torch.stack(imgs, dim=0),
        list(gt_slices),
        torch.as_tensor(labels, dtype=torch.long),
        list(diseases),
        [str(x) for x in patients],
        torch.as_tensor(slices, dtype=torch.long),
        [str(x) for x in files],
        [str(x) for x in views],
        list(concepts),
    )


def _concept_masks(
    torch, torch_f, concept_dict, names: Sequence[str], target_size, device
):
    masks = []
    present = []
    concept_dict = concept_dict or {}
    for name in names:
        raw = concept_dict.get(name)
        if raw is None:
            masks.append(torch.zeros(target_size, dtype=torch.bool, device=device))
            present.append(False)
            continue
        mask = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
        mask = mask.to(device=device, dtype=torch.float32)
        mask = mask.squeeze()
        if mask.ndim != 2:
            raise ValueError(
                f"concept mask {name!r} must reduce to (height, width), got {tuple(mask.shape)}"
            )
        if tuple(mask.shape) != tuple(target_size):
            mask = torch_f.interpolate(
                mask[None, None], size=target_size, mode="nearest"
            )[0, 0]
        binary = mask > 0.5
        masks.append(binary)
        present.append(bool(binary.any().item()))
    return torch.stack(masks), torch.as_tensor(present, dtype=torch.bool, device=device)


def exact_topk_binary_masks_numpy(
    activation_maps: np.ndarray, top_k_percent: float, eps: float = 1e-12
) -> np.ndarray:
    """Reference implementation of exact top-k masking with inactive abstention."""
    maps = np.asarray(activation_maps, dtype=np.float64)
    if maps.ndim != 3 or not 0 < top_k_percent < 1:
        raise ValueError(
            "expected (channels, height, width) maps and a fraction in (0, 1)"
        )
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, math.ceil(top_k_percent * flat.shape[1]))
    active = np.zeros_like(flat, dtype=bool)
    informative = np.ptp(flat, axis=1) > eps
    for channel in np.where(informative)[0]:
        indices = np.argsort(flat[channel], kind="stable")[-k:]
        active[channel, indices] = True
    return active.reshape(maps.shape)


def _sample_channel_concept_iou(
    torch,
    torch_f,
    activation_map,
    concept_dict,
    names: Sequence[str],
    target_size=(224, 224),
    top_k_percent=0.06,
    channel_chunk=32,
):
    """Compute per-channel IoU against each available concept without GPU OOM."""
    if not 0 < top_k_percent < 1:
        raise ValueError("top_k_percent must be in (0, 1)")
    device = activation_map.device
    masks, present = _concept_masks(
        torch, torch_f, concept_dict, names, target_size, device
    )
    n_channels = activation_map.shape[0]
    result = torch.full(
        (n_channels, len(names)), float("nan"), dtype=torch.float32, device="cpu"
    )
    if not present.any():
        return result.numpy()
    masks_present = masks[present]
    present_indices = torch.where(present)[0].cpu().numpy()
    for start in range(0, n_channels, channel_chunk):
        stop = min(start + channel_chunk, n_channels)
        up = torch_f.interpolate(
            activation_map[start:stop, None],
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        flat = up.flatten(1)
        k = max(1, math.ceil(top_k_percent * flat.shape[1]))
        informative = (flat.max(dim=1).values - flat.min(dim=1).values) > 1e-12
        top_indices = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).indices
        active_flat = torch.zeros_like(flat, dtype=torch.bool)
        active_flat.scatter_(1, top_indices, True)
        active_flat[~informative] = False
        active = active_flat.reshape_as(up)
        intersection = (active[:, None] & masks_present[None]).sum(dim=(2, 3)).float()
        union = (active[:, None] | masks_present[None]).sum(dim=(2, 3)).float()
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        block = result[start:stop]
        block[:, present_indices] = iou.detach().cpu()
        result[start:stop] = block
        del up, flat, top_indices, active_flat, active, intersection, union, iou
    return result.numpy()


def extract(args) -> list[Path]:
    if args.batch_size < 1 or args.channel_chunk < 1 or args.num_workers < 0:
        raise ValueError(
            "batch_size/channel_chunk must be positive and num_workers non-negative"
        )
    if args.max_images < 0:
        raise ValueError("max_images must be non-negative")
    if not 0 < args.top_k_percent < 1:
        raise ValueError("top_k_percent must be in (0, 1)")
    try:
        import timm
        import torch
        import torch.nn.functional as torch_f
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError(
            "Extraction requires torch, torchvision, timm, nibabel, pandas, scipy, Pillow, and numpy"
        ) from exc

    module_name, dataset_kwargs = dataset_spec_for(args.image_mode, args.concepts)
    dataset_module = importlib.import_module(module_name)
    dataset_class = dataset_module.MnMsDatasetLAX
    names = concept_names_for(args.concepts)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

    print(f"Loading dataset through {module_name}.MnMsDatasetLAX")
    print(f"Dataset kwargs: {dataset_kwargs}")
    full_dataset = dataset_class(args.mnm2_root, max_images=None, **dataset_kwargs)
    la_indices = [
        idx
        for idx, record in enumerate(full_dataset.image_slices)
        if "_LA_" in record[4]
    ]
    if args.max_images:
        la_indices = la_indices[: args.max_images]
    if not la_indices:
        raise RuntimeError("No long-axis samples were found")
    dataset = Subset(full_dataset, la_indices)
    print(f"Selected {len(dataset)} LA samples")

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=_collate,
    )
    model = _load_model(torch, timm, args.model_path, device)
    modules = dict(model.named_modules())
    requested = {block: f"features.denseblock{block}" for block in args.blocks}
    missing = [name for name in requested.values() if name not in modules]
    if missing:
        raise KeyError(f"model is missing requested layers: {missing}")
    captured = {}
    handles = []
    for block, layer_name in requested.items():
        handles.append(
            modules[layer_name].register_forward_hook(
                lambda module, inputs, output, b=block: captured.__setitem__(
                    b, output.detach()
                )
            )
        )

    activations = {block: [] for block in args.blocks}
    scores = {block: [] for block in args.blocks}
    patient_ids, sample_ids, labels, diseases, slices, files, views, logits = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    try:
        for batch_index, batch in enumerate(loader):
            (
                imgs,
                _gt,
                batch_labels,
                batch_diseases,
                batch_patients,
                batch_slices,
                batch_files,
                batch_views,
                batch_concepts,
            ) = batch
            imgs = imgs.to(device, non_blocking=device.startswith("cuda"))
            captured.clear()
            with torch.no_grad():
                if args.amp and device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(imgs)
                else:
                    output = model(imgs)
            logits.append(output.detach().float().cpu().numpy())
            for block in args.blocks:
                block_map = captured[block].float()
                activations[block].append(block_map.mean(dim=(2, 3)).cpu().numpy())
                batch_scores = []
                for sample_index in range(block_map.shape[0]):
                    batch_scores.append(
                        _sample_channel_concept_iou(
                            torch,
                            torch_f,
                            block_map[sample_index],
                            batch_concepts[sample_index],
                            names,
                            target_size=(args.image_size, args.image_size),
                            top_k_percent=args.top_k_percent,
                            channel_chunk=args.channel_chunk,
                        )
                    )
                scores[block].append(np.asarray(batch_scores, dtype=np.float32))
            patient_ids.extend(batch_patients)
            labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
            diseases.extend(batch_diseases)
            slices.extend(batch_slices.cpu().numpy().astype(int).tolist())
            files.extend(batch_files)
            views.extend(batch_views)
            sample_ids.extend(
                f"{patient}|{file}|slice={slice_idx}"
                for patient, file, slice_idx in zip(
                    batch_patients, batch_files, batch_slices.tolist()
                )
            )
            print(f"Batch {batch_index + 1}/{len(loader)} complete", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_array = np.concatenate(logits, axis=0).astype(np.float32)
    written = []
    for block in args.blocks:
        payload = {
            "pooled_activations": np.concatenate(activations[block], axis=0).astype(
                np.float32
            ),
            "concept_scores": np.concatenate(scores[block], axis=0).astype(np.float32),
            "patient_ids": np.asarray(patient_ids, dtype=str),
            "concept_names": np.asarray(names, dtype=str),
            "sample_ids": np.asarray(sample_ids, dtype=str),
            "labels": np.asarray(labels, dtype=np.int64),
            "diseases": np.asarray(diseases, dtype=str),
            "slice_indices": np.asarray(slices, dtype=np.int64),
            "file_names": np.asarray(files, dtype=str),
            "views": np.asarray(views, dtype=str),
            "logits": logits_array,
        }
        validate_payload(payload)
        metadata = {
            "schema_version": 1,
            "git_revision": _git_revision(),
            "image_mode": args.image_mode,
            "n_concepts": args.concepts,
            "block": block,
            "layer": requested[block],
            "mnm2_root": args.mnm2_root,
            "model_path": args.model_path,
            "top_k_percent": args.top_k_percent,
            "image_size": args.image_size,
            "n_samples": len(patient_ids),
            "n_patients": len(np.unique(np.asarray(patient_ids, dtype=str))),
            "dataset_module": module_name,
            "dataset_kwargs": dataset_kwargs,
            "amp": args.amp,
            "note": "Patient-level extraction artifact; no graph or semantic threshold was selected here.",
        }
        payload["metadata_json"] = np.asarray(json.dumps(metadata))
        name = f"patient_level_{args.concepts}c_{args.image_mode}_B{block}.npz"
        path = output_dir / name
        np.savez_compressed(path, **payload)
        (output_dir / name.replace(".npz", "_metadata.json")).write_text(
            json.dumps(metadata, indent=2)
        )
        print(
            f"Saved {path}: activations={payload['pooled_activations'].shape}, "
            f"concept_scores={payload['concept_scores'].shape}, patients={metadata['n_patients']}"
        )
        written.append(path)
    return written


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-mode", choices=["cropped", "full"], required=True)
    parser.add_argument("--concepts", choices=[3, 9], type=int, required=True)
    parser.add_argument(
        "--blocks", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4]
    )
    parser.add_argument("--mnm2-root", default=DEFAULT_MNM2_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="results_mnm2/patient_level")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k-percent", type=float, default=0.06)
    parser.add_argument("--channel-chunk", type=int, default=32)
    parser.add_argument(
        "--max-images", type=int, default=0, help="0 uses all LA samples"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Optional CUDA mixed precision"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    extract(args)


if __name__ == "__main__":
    main()
