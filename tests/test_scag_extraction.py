import numpy as np

from extract_scag_patient_data import (
    concept_names_for,
    dataset_spec_for,
    exact_topk_binary_masks_numpy,
    validate_payload,
)
from scag_dataset_utils import (
    dataset_paths,
    resolve_phase_matched_gt,
    stable_directory_entries,
)


def test_dataset_paths_accept_root_with_or_without_trailing_separator(tmp_path):
    data_a, labels_a = dataset_paths(tmp_path)
    data_b, labels_b = dataset_paths(str(tmp_path) + "/")
    assert data_a == data_b == str(tmp_path / "dataset")
    assert labels_a == labels_b == str(tmp_path / "dataset_information.csv")


def test_stable_directory_entries_are_sorted(tmp_path):
    for name in ["10", "02", "1"]:
        (tmp_path / name).mkdir()
    assert stable_directory_entries(tmp_path) == ["02", "1", "10"]


def test_resolve_phase_matched_gt_uses_ed_for_ed_image_and_es_for_es_image(tmp_path):
    patient = "001"
    patient_dir = tmp_path / patient
    patient_dir.mkdir()
    ed = patient_dir / f"{patient}_LA_ED_gt.nii.gz"
    es = patient_dir / f"{patient}_LA_ES_gt.nii.gz"
    ed.touch()
    es.touch()

    assert resolve_phase_matched_gt(
        tmp_path, patient, f"{patient}_LA_ED.nii.gz"
    ) == str(ed)
    assert resolve_phase_matched_gt(
        tmp_path, patient, f"{patient}_LA_ES.nii.gz"
    ) == str(es)


def test_dataset_specs_match_uploaded_loader_interfaces():
    assert dataset_spec_for("cropped", 3)[1] == {
        "crop_to_heart_union": True,
        "derive_lax_concepts": False,
    }
    assert dataset_spec_for("cropped", 9)[1] == {
        "crop_to_heart_union": True,
        "derive_lax_concepts": True,
    }
    assert dataset_spec_for("full", 3)[1] == {"get_original_concepts": True}
    assert dataset_spec_for("full", 9)[1] == {"get_original_concepts": False}


def test_concept_names_are_exact_and_ordered():
    assert concept_names_for(3) == ["LV", "MYO", "RV"]
    assert concept_names_for(9) == [
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


def test_exact_topk_masks_abstain_on_constant_channels_and_keep_exact_fraction():
    maps = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    active = exact_topk_binary_masks_numpy(maps, top_k_percent=0.25)
    assert active[0].sum() == 0
    assert active[1].sum() == 1
    assert active[1, 1, 1]
    assert active[2].sum() == 1


def test_validate_payload_accepts_patient_level_shapes_and_rejects_mismatch():
    payload = {
        "pooled_activations": np.zeros((6, 8)),
        "concept_scores": np.zeros((6, 8, 3)),
        "patient_ids": np.array(["a", "a", "b", "b", "c", "c"]),
        "concept_names": np.array(["LV", "MYO", "RV"]),
        "sample_ids": np.arange(6).astype(str),
    }
    validate_payload(payload)

    bad = dict(payload)
    bad["concept_scores"] = np.zeros((5, 8, 3))
    try:
        validate_payload(bad)
    except ValueError as exc:
        assert "sample" in str(exc).lower()
    else:
        raise AssertionError("shape mismatch should fail")
