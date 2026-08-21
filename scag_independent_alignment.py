#!/usr/bin/env python3
"""Independent, patient-level evaluation for semantic coactivation graphs.

This module intentionally does not use purity/NMI/ARI of components selected from
those same semantic labels as confirmatory evidence.  Its primary endpoint is the
held-out difference in concept-profile similarity between activation-correlation
edges and matched non-edges, with profile-permutation, topology-rewiring, and
patient-bootstrap controls.

Expected NPZ keys (created by ``extract_scag_patient_data.py``):
    pooled_activations : (samples, channels)
    concept_scores     : (samples, channels, concepts), NaN when unavailable
    patient_ids        : (samples,)
    concept_names      : (concepts,)
Optional keys are preserved only as metadata by the extractor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _nanmean(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def make_patient_folds(
    patient_ids: Sequence[str], n_folds: int, seed: int
) -> list[np.ndarray]:
    unique = np.unique(np.asarray(patient_ids).astype(str))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(unique) < n_folds:
        raise ValueError(f"need at least {n_folds} patients, found {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    return [np.asarray(x, dtype=str) for x in np.array_split(shuffled, n_folds)]


def aggregate_patient_activations(
    activations: np.ndarray,
    patient_ids: Sequence[str],
    selected_patients: Iterable[str],
) -> np.ndarray:
    """Return one equally weighted activation vector per selected patient."""
    activations = np.asarray(activations, dtype=np.float64)
    ids = np.asarray(patient_ids).astype(str)
    rows = []
    for patient in np.asarray(list(selected_patients)).astype(str):
        mask = ids == patient
        if mask.any():
            rows.append(_nanmean(activations[mask], axis=0))
    if not rows:
        raise ValueError("selected patient set has no activation samples")
    return np.asarray(rows, dtype=np.float64)


def aggregate_patient_profiles(
    concept_scores: np.ndarray,
    patient_ids: Sequence[str],
    selected_patients: Iterable[str],
) -> np.ndarray:
    """Average within patient first, then across patients to avoid slice weighting."""
    scores = np.asarray(concept_scores, dtype=np.float64)
    ids = np.asarray(patient_ids).astype(str)
    per_patient = []
    for patient in np.asarray(list(selected_patients)).astype(str):
        mask = ids == patient
        if mask.any():
            per_patient.append(_nanmean(scores[mask], axis=0))
    if not per_patient:
        raise ValueError("selected patient set has no concept-score samples")
    return _nanmean(np.asarray(per_patient), axis=0)


def normalize_profiles(raw_profiles: np.ndarray, min_profile_mass: float = 1e-8):
    raw = np.asarray(raw_profiles, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"profiles must be 2D (channels, concepts), got {raw.shape}")
    if np.nanmin(raw) < -1e-12:
        raise ValueError("concept scores must be non-negative")
    cleaned = np.where(np.isfinite(raw), np.maximum(raw, 0.0), 0.0)
    has_any = np.isfinite(raw).any(axis=1)
    magnitude = cleaned.sum(axis=1)
    valid = has_any & np.isfinite(magnitude) & (magnitude >= min_profile_mass)
    profiles = np.full_like(cleaned, np.nan, dtype=np.float64)
    profiles[valid] = cleaned[valid] / magnitude[valid, None]
    return profiles, magnitude, valid


def spearman_correlation(
    samples_by_channel: np.ndarray, device: str = "cpu", chunk_size: int = 0
):
    """Tie-correct Spearman correlation; optional CUDA only accelerates matmul.

    Ranking is deliberately performed with SciPy's average ranks.  The original
    GPU ordinal-ranking implementation breaks ties arbitrarily, which is unsafe
    for sparse/ReLU activations.
    """
    x = np.asarray(samples_by_channel, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("need a 2D (samples, channels) array with at least 3 samples")
    ranks = rankdata(x, axis=0, method="average")
    ranks -= ranks.mean(axis=0, keepdims=True)
    sd = ranks.std(axis=0, ddof=1)
    valid = np.isfinite(sd) & (sd > 0)
    z = np.zeros_like(ranks, dtype=np.float64)
    z[:, valid] = ranks[:, valid] / sd[valid]
    denom = x.shape[0] - 1

    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA requested, but PyTorch is not installed") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        tensor = torch.as_tensor(z, dtype=torch.float64, device=device)
        n = tensor.shape[1]
        if chunk_size and n > chunk_size:
            corr = np.empty((n, n), dtype=np.float64)
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                corr[start:stop] = (
                    ((tensor[:, start:stop].T @ tensor) / denom).cpu().numpy()
                )
        else:
            corr = ((tensor.T @ tensor) / denom).cpu().numpy()
        del tensor
    else:
        corr = (z.T @ z) / denom

    corr[~valid, :] = np.nan
    corr[:, ~valid] = np.nan
    corr[valid, valid] = 1.0
    return np.clip(corr, -1.0, 1.0)


def build_fixed_density_graph(corr: np.ndarray, density: float):
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation matrix must be square")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    n = corr.shape[0]
    u, v = np.triu_indices(n, k=1)
    weights = corr[u, v]
    positive = np.isfinite(weights) & (weights > 0)
    target = math.ceil(density * len(weights))
    candidate = np.where(positive)[0]
    if candidate.size:
        order = candidate[np.argsort(weights[candidate], kind="stable")[::-1]]
        selected = order[: min(target, len(order))]
    else:
        selected = np.array([], dtype=np.int64)
    edge_u, edge_v, edge_w = u[selected], v[selected], weights[selected]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        (int(a), int(b), float(w)) for a, b, w in zip(edge_u, edge_v, edge_w)
    )
    return (
        graph,
        edge_u.astype(np.int64),
        edge_v.astype(np.int64),
        edge_w.astype(np.float64),
    )


def degree_preserving_rewire(
    graph: nx.Graph, seed: int, swap_factor: int = 10
) -> nx.Graph:
    rewired = nx.Graph()
    rewired.add_nodes_from(graph.nodes())
    rewired.add_edges_from(graph.edges())
    m = rewired.number_of_edges()
    if m < 2 or rewired.number_of_nodes() < 4:
        rewired.graph["requested_swaps"] = 0
        rewired.graph["edge_change_fraction"] = 0.0
        return rewired
    requested = max(1, int(swap_factor * m))
    original_edges = {tuple(sorted(edge)) for edge in rewired.edges()}
    try:
        nx.double_edge_swap(
            rewired, nswap=requested, max_tries=max(100, requested * 30), seed=seed
        )
    except (nx.NetworkXAlgorithmError, nx.NetworkXError) as exc:
        raise RuntimeError(f"degree-preserving rewiring failed: {exc}") from exc
    changed_edges = original_edges.symmetric_difference(
        {tuple(sorted(edge)) for edge in rewired.edges()}
    )
    rewired.graph["requested_swaps"] = requested
    rewired.graph["edge_change_fraction"] = len(changed_edges) / max(2 * m, 1)
    return rewired


def js_similarity_pairs(
    profiles: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    p = profiles[np.asarray(u, dtype=np.int64)]
    q = profiles[np.asarray(v, dtype=np.int64)]
    m = 0.5 * (p + q)

    def kl(a, b):
        positive = a > 0
        terms = np.zeros_like(a, dtype=np.float64)
        terms[positive] = a[positive] * (np.log2(a[positive]) - np.log2(b[positive]))
        return terms.sum(axis=1)

    divergence = np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0)
    return 1.0 - np.sqrt(divergence)


def _quantile_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.all(values == values[0]):
        return np.zeros(len(values), dtype=np.int64)
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) <= 2:
        return np.zeros(len(values), dtype=np.int64)
    return np.digitize(values, quantiles[1:-1], right=True).astype(np.int64)


def matched_nonedges(
    graph: nx.Graph,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    valid_nodes: np.ndarray,
    node_activity: np.ndarray,
    seed: int,
):
    """Match non-edges by endpoint degree and train-set activation bins."""
    rng = np.random.default_rng(seed)
    valid_nodes = np.asarray(valid_nodes, dtype=np.int64)
    degree = np.array(
        [graph.degree(i) for i in range(graph.number_of_nodes())], dtype=np.float64
    )
    degree_bin = _quantile_bins(degree, n_bins=3)
    activity_bin = _quantile_bins(np.asarray(node_activity, dtype=np.float64), n_bins=3)

    node_class = degree_bin * (int(activity_bin.max()) + 1) + activity_bin
    n_classes = int(node_class.max()) + 1

    def pair_class(a, b):
        left = np.minimum(node_class[a], node_class[b])
        right = np.maximum(node_class[a], node_class[b])
        return left * n_classes + right

    n_nodes = graph.number_of_nodes()
    edge_pairs = np.asarray(list(graph.edges()), dtype=np.int64)
    if edge_pairs.size:
        edge_left = np.minimum(edge_pairs[:, 0], edge_pairs[:, 1])
        edge_right = np.maximum(edge_pairs[:, 0], edge_pairs[:, 1])
        edge_codes = edge_left * n_nodes + edge_right
    else:
        edge_codes = np.array([], dtype=np.int64)

    local_u, local_v = np.triu_indices(len(valid_nodes), k=1)
    candidate_u, candidate_v = valid_nodes[local_u], valid_nodes[local_v]
    candidate_codes = candidate_u * n_nodes + candidate_v
    is_edge = np.isin(candidate_codes, edge_codes, assume_unique=False)
    non_u, non_v = candidate_u[~is_edge], candidate_v[~is_edge]
    if not len(non_u):
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.zeros(len(edge_u), dtype=bool),
            0.0,
        )

    non_keys = pair_class(non_u, non_v)
    order = np.argsort(non_keys, kind="stable")
    sorted_keys = non_keys[order]
    sorted_u, sorted_v = non_u[order], non_v[order]
    edge_keys = pair_class(
        np.asarray(edge_u, dtype=np.int64), np.asarray(edge_v, dtype=np.int64)
    )

    matched = np.empty((len(edge_keys), 2), dtype=np.int64)
    keep = np.zeros(len(edge_keys), dtype=bool)
    exact = 0
    for index, key in enumerate(edge_keys):
        start = int(np.searchsorted(sorted_keys, key, side="left"))
        stop = int(np.searchsorted(sorted_keys, key, side="right"))
        if stop > start:
            selected = int(rng.integers(start, stop))
            matched[index] = sorted_u[selected], sorted_v[selected]
            keep[index] = True
            exact += 1
    return matched[keep, 0], matched[keep, 1], keep, exact / max(len(matched), 1)


def _mean_or_nan(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return (
        float(np.nanmean(values))
        if values.size and np.isfinite(values).any()
        else float("nan")
    )


def _null_summary(observed: float, null_values: Sequence[float]):
    null = np.asarray(null_values, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not len(null):
        return float("nan"), float("nan"), float("nan")
    p = (1 + int(np.sum(null >= observed))) / (len(null) + 1)
    return (
        float(null.mean()),
        float(null.std(ddof=1) if len(null) > 1 else 0.0),
        float(p),
    )


def _bootstrap_ci(values: Sequence[float], alpha: float = 0.05):
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(clean, [alpha / 2, 1 - alpha / 2]))


def _global_metrics(
    graph: nx.Graph, profiles: np.ndarray, valid: np.ndarray, seed: int
):
    labels = np.argmax(profiles[valid], axis=1)
    valid_nodes = np.where(valid)[0]
    sub = graph.subgraph(valid_nodes).copy()
    if sub.number_of_edges() == 0 or len(valid_nodes) < 2:
        return float("nan"), float("nan"), 0
    communities = nx.community.louvain_communities(sub, weight="weight", seed=seed)
    membership = {}
    for cid, community in enumerate(communities):
        for node in community:
            membership[int(node)] = cid
    predicted = np.array([membership[int(node)] for node in valid_nodes])
    return (
        float(normalized_mutual_info_score(labels, predicted)),
        float(adjusted_rand_score(labels, predicted)),
        len(communities),
    )


def _dose_response(
    corr: np.ndarray, profiles: np.ndarray, valid: np.ndarray, fold: int
):
    nodes = np.where(valid)[0]
    u0, v0 = np.triu_indices(len(nodes), k=1)
    u, v = nodes[u0], nodes[v0]
    coupling = corr[u, v]
    semantic = js_similarity_pairs(profiles, u, v)
    keep = np.isfinite(coupling) & np.isfinite(semantic)
    coupling, semantic = coupling[keep], semantic[keep]
    if not len(coupling):
        return [], float("nan")
    order = np.argsort(coupling)
    bins = np.array_split(order, min(10, len(order)))
    rows = [
        {
            "fold": fold,
            "coupling_bin": idx + 1,
            "n_pairs": len(sel),
            "mean_coupling": float(coupling[sel].mean()),
            "mean_semantic_similarity": float(semantic[sel].mean()),
        }
        for idx, sel in enumerate(bins)
        if len(sel)
    ]
    rho = (
        float(spearmanr(coupling, semantic).statistic)
        if len(coupling) >= 3
        else float("nan")
    )
    return rows, rho


def run_crossfit_analysis(
    pooled_activations: np.ndarray,
    concept_scores: np.ndarray,
    patient_ids: Sequence[str],
    concept_names: Sequence[str],
    output_dir: os.PathLike | str,
    densities: Sequence[float] = (0.01, 0.025, 0.05, 0.10),
    n_folds: int = 5,
    n_permutations: int = 1000,
    n_rewires: int = 200,
    rewire_swap_factor: int = 1,
    n_bootstraps: int = 1000,
    min_profile_mass: float = 1e-8,
    seed: int = 42,
    device: str = "cpu",
    chunk_size: int = 0,
    min_match_rate: float = 0.7,
):
    activations = np.asarray(pooled_activations, dtype=np.float64)
    scores = np.asarray(concept_scores, dtype=np.float64)
    ids = np.asarray(patient_ids).astype(str)
    names = [str(x) for x in concept_names]
    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative")
    if n_permutations < 1 or n_rewires < 1 or n_bootstraps < 1:
        raise ValueError("permutations, rewires, and bootstraps must be positive")
    if rewire_swap_factor < 1:
        raise ValueError("rewire_swap_factor must be positive")
    if min_profile_mass < 0:
        raise ValueError("min_profile_mass must be non-negative")
    if not 0 < min_match_rate <= 1:
        raise ValueError("min_match_rate must be in (0, 1]")
    if activations.ndim != 2:
        raise ValueError("pooled_activations must have shape (samples, channels)")
    if scores.ndim != 3:
        raise ValueError("concept_scores must have shape (samples, channels, concepts)")
    if (
        activations.shape[:1] != scores.shape[:1]
        or activations.shape[1] != scores.shape[1]
    ):
        raise ValueError(
            "activation and concept-score sample/channel dimensions disagree"
        )
    if len(ids) != len(activations):
        raise ValueError("patient_ids length does not equal sample count")
    if scores.shape[2] != len(names):
        raise ValueError("concept_names length does not equal concept-score dimension")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds = make_patient_folds(ids, n_folds=n_folds, seed=seed)
    all_patients = set(np.unique(ids).tolist())
    rng = np.random.default_rng(seed)
    metric_rows = []
    dose_rows = []
    permutation_effects_by_density = {float(d): [] for d in densities}
    bootstrap_effects_by_density = {float(d): [] for d in densities}

    for fold_index, eval_patients in enumerate(folds):
        eval_set = set(eval_patients.tolist())
        train_patients = np.asarray(sorted(all_patients - eval_set), dtype=str)
        train_activations = aggregate_patient_activations(
            activations, ids, train_patients
        )
        corr = spearman_correlation(
            train_activations, device=device, chunk_size=chunk_size
        )
        node_activity = np.nanstd(train_activations, axis=0, ddof=1)
        raw_profiles = aggregate_patient_profiles(scores, ids, eval_patients)
        profiles, profile_mass, valid = normalize_profiles(
            raw_profiles, min_profile_mass
        )
        fold_dose, pair_rho = _dose_response(corr, profiles, valid, fold_index)
        dose_rows.extend(fold_dose)

        for density in densities:
            graph, edge_u, edge_v, edge_w = build_fixed_density_graph(
                corr, float(density)
            )
            edge_valid = valid[edge_u] & valid[edge_v]
            eu, ev, ew = edge_u[edge_valid], edge_v[edge_valid], edge_w[edge_valid]
            topology_observed_edge_similarity = _mean_or_nan(
                js_similarity_pairs(profiles, eu, ev)
            )
            valid_nodes = np.where(valid)[0]
            non_u, non_v, match_keep, exact_rate = matched_nonedges(
                graph,
                eu,
                ev,
                valid_nodes,
                node_activity,
                seed + fold_index * 1009 + int(density * 1e5),
            )
            if exact_rate < min_match_rate:
                raise RuntimeError(
                    f"only {exact_rate:.1%} of edges had exact degree/activity-stratum "
                    f"non-edge matches (required {min_match_rate:.1%}); use a lower density "
                    "or explicitly lower --min-match-rate"
                )
            eu, ev, ew = eu[match_keep], ev[match_keep], ew[match_keep]
            edge_similarity = (
                js_similarity_pairs(profiles, eu, ev) if len(eu) else np.array([])
            )
            non_similarity = (
                js_similarity_pairs(profiles, non_u, non_v)
                if len(non_u)
                else np.array([])
            )
            observed = _mean_or_nan(edge_similarity)
            non_mean = _mean_or_nan(non_similarity)
            effect = observed - non_mean

            perm_null_effects = []
            for _ in range(n_permutations):
                permuted = profiles.copy()
                permuted[valid_nodes] = profiles[rng.permutation(valid_nodes)]
                perm_null_effects.append(
                    _mean_or_nan(js_similarity_pairs(permuted, eu, ev))
                    - _mean_or_nan(js_similarity_pairs(permuted, non_u, non_v))
                )
            permutation_effects_by_density[float(density)].append(
                np.asarray(perm_null_effects, dtype=np.float64)
            )
            perm_mean, perm_sd, perm_p = _null_summary(effect, perm_null_effects)

            topology_null = []
            topology_changes = []
            topology_failures = 0
            valid_graph = graph.subgraph(valid_nodes).copy()
            for rep in range(n_rewires):
                try:
                    rewired = degree_preserving_rewire(
                        valid_graph,
                        seed + 100_000 * (fold_index + 1) + rep,
                        swap_factor=rewire_swap_factor,
                    )
                except RuntimeError:
                    topology_failures += 1
                    continue
                change = float(rewired.graph.get("edge_change_fraction", 0.0))
                if change < 0.05:
                    topology_failures += 1
                    continue
                pairs = np.asarray(list(rewired.edges()), dtype=np.int64)
                if pairs.size:
                    topology_changes.append(change)
                    topology_null.append(
                        _mean_or_nan(
                            js_similarity_pairs(profiles, pairs[:, 0], pairs[:, 1])
                        )
                    )
            minimum_success = max(1, math.ceil(0.8 * n_rewires))
            if len(topology_null) < minimum_success:
                raise RuntimeError(
                    f"only {len(topology_null)}/{n_rewires} topology nulls rewired "
                    "successfully with >=5% edge change"
                )
            topo_mean, topo_sd, topo_p = _null_summary(
                topology_observed_edge_similarity, topology_null
            )

            bootstrap_effects = []
            eval_array = np.asarray(eval_patients, dtype=str)
            for _ in range(n_bootstraps):
                sampled = rng.choice(eval_array, size=len(eval_array), replace=True)
                boot_raw = aggregate_patient_profiles(scores, ids, sampled)
                boot_profiles, _, boot_valid = normalize_profiles(
                    boot_raw, min_profile_mass
                )
                keep_e = boot_valid[eu] & boot_valid[ev]
                keep_n = boot_valid[non_u] & boot_valid[non_v]
                boot_effect = float("nan")
                if keep_e.any() and keep_n.any():
                    boot_effect = _mean_or_nan(
                        js_similarity_pairs(boot_profiles, eu[keep_e], ev[keep_e])
                    ) - _mean_or_nan(
                        js_similarity_pairs(boot_profiles, non_u[keep_n], non_v[keep_n])
                    )
                bootstrap_effects.append(boot_effect)
            bootstrap_effects_by_density[float(density)].append(
                np.asarray(bootstrap_effects, dtype=np.float64)
            )
            global_nmi, global_ari, n_communities = _global_metrics(
                graph, profiles, valid, seed + fold_index
            )

            metric_rows.append(
                {
                    "fold": fold_index,
                    "density": float(density),
                    "n_train_patients": len(train_patients),
                    "n_eval_patients": len(eval_patients),
                    "n_channels": activations.shape[1],
                    "n_valid_profile_channels": int(valid.sum()),
                    "profile_channel_coverage": float(valid.mean()),
                    "n_graph_edges": graph.number_of_edges(),
                    "n_valid_edges": len(eu),
                    "mean_edge_weight": _mean_or_nan(ew),
                    "edge_semantic_similarity": observed,
                    "matched_nonedge_semantic_similarity": non_mean,
                    "edge_minus_matched_nonedge": effect,
                    "matched_exact_stratum_rate": exact_rate,
                    "primary_permutation_effect_mean": perm_mean,
                    "primary_permutation_effect_sd": perm_sd,
                    "primary_permutation_p_fold": perm_p,
                    "topology_edge_similarity_mean_secondary": topo_mean,
                    "topology_edge_similarity_sd_secondary": topo_sd,
                    "topology_edge_similarity_p_fold_secondary": topo_p,
                    "topology_observed_edge_similarity_secondary": topology_observed_edge_similarity,
                    "topology_successful_rewires": len(topology_null),
                    "topology_failed_rewires": topology_failures,
                    "topology_mean_edge_change_fraction": _mean_or_nan(
                        topology_changes
                    ),
                    "pairwise_coupling_semantic_rho_descriptive": pair_rho,
                    "global_nmi": global_nmi,
                    "global_ari": global_ari,
                    "n_communities": n_communities,
                    "mean_profile_mass": _mean_or_nan(profile_mass[valid]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    dose = pd.DataFrame(dose_rows)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    dose.to_csv(output / "dose_response.csv", index=False)

    density_summary = {}
    for density, group in metrics.groupby("density"):
        density_key = float(density)
        effects = group["edge_minus_matched_nonedge"].to_numpy(dtype=float)
        observed_effect = float(np.nanmean(effects))
        permutation_matrix = np.vstack(permutation_effects_by_density[density_key])
        aggregate_permutation = np.nanmean(permutation_matrix, axis=0)
        perm_mean, perm_sd, perm_p = _null_summary(
            observed_effect, aggregate_permutation
        )
        bootstrap_matrix = np.vstack(bootstrap_effects_by_density[density_key])
        aggregate_bootstrap = np.nanmean(bootstrap_matrix, axis=0)
        ci_low, ci_high = _bootstrap_ci(aggregate_bootstrap)
        density_summary[str(density_key)] = {
            "n_folds": len(group),
            "mean_edge_semantic_similarity": float(
                group["edge_semantic_similarity"].mean()
            ),
            "mean_matched_nonedge_semantic_similarity": float(
                group["matched_nonedge_semantic_similarity"].mean()
            ),
            "mean_edge_minus_matched_nonedge": observed_effect,
            "patient_bootstrap_ci_low": ci_low,
            "patient_bootstrap_ci_high": ci_high,
            "primary_profile_permutation_null_mean": perm_mean,
            "primary_profile_permutation_null_sd": perm_sd,
            "primary_profile_permutation_p": perm_p,
            "mean_exact_match_rate": float(group["matched_exact_stratum_rate"].mean()),
            "mean_topology_edge_change_fraction": float(
                group["topology_mean_edge_change_fraction"].mean()
            ),
            "mean_global_nmi": float(group["global_nmi"].mean()),
            "mean_global_ari": float(group["global_ari"].mean()),
        }
    summary = {
        "design": "patient-level cross-fit: graph on train patients; profiles on held-out patients",
        "primary_endpoint": "edge semantic similarity minus train-statistic-matched non-edge similarity",
        "n_samples": len(activations),
        "n_patients": len(np.unique(ids)),
        "n_channels": int(activations.shape[1]),
        "concept_names": names,
        "densities": density_summary,
        "warnings": [
            "This controls extraction/evaluation reuse but cannot undo classifier-training leakage without the classifier split file.",
            "Pairwise coupling-semantic rho is descriptive because channel pairs share nodes.",
            "Global NMI/ARI are descriptive and are not evidence for semantically filtered components.",
            "The patient bootstrap resamples held-out patients within folds and aggregates the same replicate across folds; graph estimates remain fixed within each fold.",
            "Topology rewiring tests edge similarity as a secondary null; the declared primary endpoint is tested by the aggregate profile-permutation contrast.",
        ],
    }
    config = {
        "densities": [float(x) for x in densities],
        "n_folds": n_folds,
        "n_permutations": n_permutations,
        "n_rewires": n_rewires,
        "rewire_swap_factor": rewire_swap_factor,
        "n_bootstraps": n_bootstraps,
        "min_profile_mass": min_profile_mass,
        "min_match_rate": min_match_rate,
        "seed": seed,
        "device": device,
        "chunk_size": chunk_size,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2))
    return {"summary": summary, "metrics": metrics, "dose_response": dose}


def generate_synthetic_dataset(
    seed: int = 0,
    aligned: bool = True,
    n_patients: int = 40,
    samples_per_patient: int = 3,
    n_channels: int = 24,
    n_concepts: int = 3,
):
    rng = np.random.default_rng(seed)
    patient_ids = np.repeat(
        [f"P{i:03d}" for i in range(n_patients)], samples_per_patient
    )
    n_samples = len(patient_ids)
    group = np.arange(n_channels) % n_concepts
    latent = rng.normal(size=(n_samples, n_concepts))
    patient_offset = rng.normal(scale=0.25, size=(n_patients, n_concepts))
    latent += np.repeat(patient_offset, samples_per_patient, axis=0)
    if aligned:
        activations = np.column_stack(
            [
                latent[:, group[ch]] + rng.normal(scale=0.35, size=n_samples)
                for ch in range(n_channels)
            ]
        )
    else:
        activations = rng.normal(size=(n_samples, n_channels))
    base = np.eye(n_concepts)[group]
    concept_scores = np.empty((n_samples, n_channels, n_concepts), dtype=np.float64)
    for sample in range(n_samples):
        noise = rng.gamma(shape=1.0, scale=0.03, size=(n_channels, n_concepts))
        strength = rng.uniform(0.7, 1.0, size=(n_channels, 1))
        concept_scores[sample] = strength * base + noise
    return {
        "pooled_activations": activations,
        "concept_scores": concept_scores,
        "patient_ids": patient_ids,
        "concept_names": np.array([f"C{i}" for i in range(n_concepts)]),
    }


def _load_npz(path: os.PathLike | str):
    with np.load(path, allow_pickle=False) as data:
        required = [
            "pooled_activations",
            "concept_scores",
            "patient_ids",
            "concept_names",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} is missing required keys: {missing}")
        return {key: np.asarray(data[key]) for key in required}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, help="Patient-level NPZ from extract_scag_patient_data.py"
    )
    source.add_argument(
        "--simulate",
        choices=["aligned", "null", "both"],
        help="Run a complete synthetic verification without medical data",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--densities", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.10]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--rewires", type=int, default=200)
    parser.add_argument(
        "--rewire-swap-factor",
        type=int,
        default=1,
        help="Attempt this many degree-preserving swaps per graph edge for each topology null",
    )
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--min-profile-mass", type=float, default=1e-8)
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=0.7,
        help="Fail if fewer than this fraction of edges have exact degree/activity-stratum non-edge matches",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small null/bootstrap counts for a smoke test",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    permutations = min(args.permutations, 50) if args.quick else args.permutations
    rewires = min(args.rewires, 20) if args.quick else args.rewires
    bootstraps = min(args.bootstraps, 50) if args.quick else args.bootstraps

    def run(data, out):
        result = run_crossfit_analysis(
            **data,
            output_dir=out,
            densities=args.densities,
            n_folds=args.folds,
            n_permutations=permutations,
            n_rewires=rewires,
            rewire_swap_factor=args.rewire_swap_factor,
            n_bootstraps=bootstraps,
            min_profile_mass=args.min_profile_mass,
            min_match_rate=args.min_match_rate,
            seed=args.seed,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(result["summary"], indent=2))

    if args.input:
        run(_load_npz(args.input), args.output)
    elif args.simulate == "both":
        run(
            generate_synthetic_dataset(seed=args.seed, aligned=True),
            args.output / "aligned",
        )
        run(
            generate_synthetic_dataset(seed=args.seed, aligned=False),
            args.output / "null",
        )
    else:
        run(
            generate_synthetic_dataset(
                seed=args.seed, aligned=args.simulate == "aligned"
            ),
            args.output,
        )


if __name__ == "__main__":
    main()
