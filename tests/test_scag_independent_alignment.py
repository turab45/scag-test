import json

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from scag_independent_alignment import (
    _load_npz,
    aggregate_patient_profiles,
    build_fixed_density_graph,
    degree_preserving_rewire,
    generate_synthetic_dataset,
    make_patient_folds,
    matched_nonedges,
    normalize_profiles,
    run_crossfit_analysis,
    spearman_correlation,
)


def test_patient_folds_are_deterministic_disjoint_and_complete():
    patient_ids = np.array(["p1", "p1", "p2", "p3", "p3", "p4", "p5", "p6"])
    folds_a = make_patient_folds(patient_ids, n_folds=3, seed=17)
    folds_b = make_patient_folds(patient_ids, n_folds=3, seed=17)

    assert [x.tolist() for x in folds_a] == [x.tolist() for x in folds_b]
    flattened = np.concatenate(folds_a)
    assert sorted(flattened.tolist()) == sorted(np.unique(patient_ids).tolist())
    assert (
        sum(
            len(set(a) & set(b))
            for i, a in enumerate(folds_a)
            for b in folds_a[i + 1 :]
        )
        == 0
    )


def test_patient_profiles_weight_patients_equally_not_slices():
    # p1 has three high-valued slices; p2 has one zero-valued slice.
    scores = np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
    patient_ids = np.array(["p1", "p1", "p1", "p2"])

    profile = aggregate_patient_profiles(scores, patient_ids, ["p1", "p2"])

    np.testing.assert_allclose(profile, np.array([[0.5, 0.5]]))


def test_normalize_profiles_excludes_zero_evidence_channels():
    raw = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    profiles, magnitude, valid = normalize_profiles(raw, min_profile_mass=1e-8)

    np.testing.assert_allclose(profiles[0], [0.5, 0.25, 0.25])
    assert valid.tolist() == [True, False, False]
    assert magnitude[0] == 4.0
    assert np.isnan(profiles[1:]).all()


def test_spearman_uses_average_ranks_for_ties():
    samples = np.array(
        [
            [0.0, 4.0, 1.0],
            [0.0, 3.0, 2.0],
            [1.0, 2.0, 2.0],
            [1.0, 1.0, 4.0],
            [2.0, 0.0, 5.0],
        ]
    )
    observed = spearman_correlation(samples, device="cpu")
    expected = spearmanr(samples, axis=0).statistic

    np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)


def test_fixed_density_graph_selects_strongest_positive_edges():
    corr = np.array(
        [
            [1.0, 0.9, 0.8, -0.1],
            [0.9, 1.0, 0.7, 0.2],
            [0.8, 0.7, 1.0, 0.1],
            [-0.1, 0.2, 0.1, 1.0],
        ]
    )
    graph, edge_u, edge_v, edge_w = build_fixed_density_graph(corr, density=0.5)

    assert graph.number_of_edges() == 3  # ceil(0.5 * 6 possible pairs)
    assert set(np.round(edge_w, 6)) == {0.9, 0.8, 0.7}
    assert set(zip(edge_u.tolist(), edge_v.tolist())) == {(0, 1), (0, 2), (1, 2)}


def test_degree_preserving_rewire_preserves_degree_sequence():
    graph = nx.cycle_graph(10)
    graph.add_edges_from([(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
    rewired = degree_preserving_rewire(graph, seed=4, swap_factor=3)

    assert sorted(dict(graph.degree()).values()) == sorted(
        dict(rewired.degree()).values()
    )
    assert graph.number_of_edges() == rewired.number_of_edges()
    assert rewired.graph["edge_change_fraction"] > 0


def test_matched_nonedges_return_only_exact_stratum_matches():
    graph = nx.cycle_graph(8)
    edge_u = np.array([0, 1, 2, 3])
    edge_v = np.array([1, 2, 3, 4])
    non_u, non_v, keep, rate = matched_nonedges(
        graph, edge_u, edge_v, np.arange(8), np.ones(8), seed=7
    )
    assert len(non_u) == len(non_v) == int(keep.sum())
    assert set(zip(non_u.tolist(), non_v.tolist())).isdisjoint(
        {tuple(sorted(edge)) for edge in graph.edges()}
    )
    assert rate == keep.mean()


def test_npz_loader_rejects_object_pickle_arrays(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(
        path,
        pooled_activations=np.zeros((2, 2)),
        concept_scores=np.zeros((2, 2, 1)),
        patient_ids=np.array([{"patient": 1}, {"patient": 2}], dtype=object),
        concept_names=np.array(["C0"]),
    )
    try:
        _load_npz(path)
    except ValueError as exc:
        assert "object" in str(exc).lower() or "pickle" in str(exc).lower()
    else:
        raise AssertionError("object arrays must not be deserialized")


def test_aligned_simulation_has_larger_effect_than_null_simulation(tmp_path):
    aligned = generate_synthetic_dataset(
        seed=3, aligned=True, n_patients=30, samples_per_patient=3
    )
    null = generate_synthetic_dataset(
        seed=3, aligned=False, n_patients=30, samples_per_patient=3
    )

    aligned_result = run_crossfit_analysis(
        **aligned,
        output_dir=tmp_path / "aligned",
        densities=[0.10],
        n_folds=3,
        n_permutations=40,
        n_rewires=20,
        n_bootstraps=40,
        seed=11,
        device="cpu",
    )
    null_result = run_crossfit_analysis(
        **null,
        output_dir=tmp_path / "null",
        densities=[0.10],
        n_folds=3,
        n_permutations=40,
        n_rewires=20,
        n_bootstraps=40,
        seed=11,
        device="cpu",
    )

    aligned_effect = aligned_result["summary"]["densities"]["0.1"][
        "mean_edge_minus_matched_nonedge"
    ]
    null_effect = null_result["summary"]["densities"]["0.1"][
        "mean_edge_minus_matched_nonedge"
    ]
    assert aligned_effect > 0.08
    assert aligned_effect > null_effect + 0.05
    primary = aligned_result["summary"]["densities"]["0.1"]
    assert primary["patient_bootstrap_ci_low"] > 0
    assert primary["primary_profile_permutation_p"] < 0.05

    for dirname in ("aligned", "null"):
        root = tmp_path / dirname
        assert (root / "fold_metrics.csv").exists()
        assert (root / "dose_response.csv").exists()
        assert (root / "summary.json").exists()
        json.loads((root / "summary.json").read_text())


def test_negative_chunk_size_is_rejected(tmp_path):
    data = generate_synthetic_dataset(seed=9, aligned=True, n_patients=12)
    try:
        run_crossfit_analysis(
            **data,
            output_dir=tmp_path,
            densities=[0.1],
            n_folds=3,
            n_permutations=5,
            n_rewires=5,
            n_bootstraps=5,
            chunk_size=-1,
        )
    except ValueError as exc:
        assert "chunk" in str(exc).lower()
    else:
        raise AssertionError("negative chunk size should fail")
