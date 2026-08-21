"""Shared safeguards for the uploaded M&Ms-2 dataset loaders."""

from __future__ import annotations

import os
import re
from pathlib import Path


def dataset_paths(root: os.PathLike | str) -> tuple[str, str]:
    root_path = Path(root)
    return str(root_path / "dataset"), str(root_path / "dataset_information.csv")


def stable_directory_entries(path: os.PathLike | str) -> list[str]:
    return sorted(os.listdir(path))


def resolve_phase_matched_gt(
    dataset_dir: os.PathLike | str, patient_folder: str, file_name: str
) -> str | None:
    """Resolve the GT file matching both view and cardiac phase.

    The original loaders returned the first existing ED/ES mask, so an ES image
    could silently receive the ED mask.  This resolver requires a matching phase
    when the image filename contains ED or ES.
    """
    name = str(file_name)
    view_match = re.search(r"_(SA|LA)_", name, flags=re.IGNORECASE)
    if not view_match:
        return None
    view = view_match.group(1).upper()
    phase_match = re.search(r"_(ED|ES)(?:_|\.|$)", name, flags=re.IGNORECASE)
    phases = [phase_match.group(1).upper()] if phase_match else ["ED", "ES"]
    patient_dir = Path(dataset_dir) / str(patient_folder)
    for phase in phases:
        candidate = patient_dir / f"{patient_folder}_{view}_{phase}_gt.nii.gz"
        if candidate.exists():
            return str(candidate)
    return None
