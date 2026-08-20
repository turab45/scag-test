#!/usr/bin/env python3
"""
coactivation_analysis_3concepts_fullimg.py
===========================================

Cardiac DenseNet-161 coactivation study: build + analyse.
**3-concept (LV / MYO / RV) FULL-IMAGE variant.**

This is the SAME pipeline as the cropped `coactivation_analysis.py`, but the
neuron-concept association (NCA) matrix is built over the three GT LA concepts
on the FULL (uncropped) image. The only thing that differs from the cropped run
is the dataset: this variant uses the full-image dataset module exactly as in
your full-image notebook --

    from MnMs2DatasetConcpets import MnMsDatasetLAX
    MnMsDatasetLAX(root, max_images=None, get_original_concepts=True)

-- instead of MnMs2DatasetConcpetsHeartUnion(... crop_to_heart_union=True).

Everything downstream -- pooled + spatial graph construction, GPU/CPU Spearman,
Louvain + PageRank, purity/NMI/ARI, hard-argmax and JSD homophily with
permutation nulls + assortativity, edge/node/combined substructures, and the
on-graph substructure plots -- is byte-for-byte the cropped pipeline (already
concept-count-agnostic: it reads concept_names from the npz and uses
rows.shape[1] throughout).

All configuration lives in main() -- edit the variables there. No CLI args.
The only differences from the cropped script are:
  * the dataset module + args above (full image, not heart-union-cropped),
  * cfg.image_tag = 'fullimg', so graph/npz filenames carry "fullimg" instead
    of "cropped" and never collide with the cropped artifacts.

STAGE 1 (BUILD=True)  Forward DenseNet-161 over the LA slices, accumulate the
                      NCA matrix, and build BOTH the pooled and spatial
                      coactivation graphs.

STAGE 2 (always)      Load both graphs + the .npz and run the full analysis.

Everything prints progress with elapsed-time stamps so you always see what's running.
"""

import os
import re
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from my_coactivation_graph import CoactivationGraphBuilder

try:
    from networkx.algorithms.community import louvain_communities
except Exception:
    louvain_communities = None

# ---------------------------------------------------------------- logging
_START = time.time()


def log(msg):
    print(f"[{time.time() - _START:7.1f}s] {msg}", flush=True)


def banner(msg):
    print(f"\n{'='*72}\n{msg}\n{'='*72}", flush=True)


# ======================================================================
# GPU Spearman (the bottleneck, moved to CUDA)
# ======================================================================

def gpu_spearman_from_samples(samples_by_neuron, device, chunk_neurons=0):
    """Spearman correlation between neurons, computed on the GPU.

    samples_by_neuron : np.ndarray (S, N_neurons)  -- S = samples (images, or
                        foreground cells for the spatial graph).
    Returns corr (N_neurons, N_neurons) float32 on CPU.

    Ranks are ordinal (ties broken arbitrarily); for continuous activations this
    matches average-rank Spearman to ~1e-6 and is far faster.
    """
    import torch
    log(f"  [spearman] moving {samples_by_neuron.shape} to {device} and ranking ...")
    t = torch.as_tensor(np.ascontiguousarray(samples_by_neuron), device=device, dtype=torch.float32)
    S, N = t.shape
    ranks = t.argsort(dim=0).argsort(dim=0).to(torch.float32)
    del t
    ranks -= ranks.mean(dim=0, keepdim=True)
    std = ranks.std(dim=0, unbiased=True, keepdim=True)
    std[std == 0] = 1.0
    ranks /= std
    log(f"  [spearman] correlating {N} neurons over {S} samples (GPU matmul) ...")
    denom = max(S - 1, 1)
    if chunk_neurons and N > chunk_neurons:
        corr = torch.empty((N, N), device='cpu', dtype=torch.float32)
        for a in range(0, N, chunk_neurons):
            b = min(a + chunk_neurons, N)
            block = (ranks[:, a:b].T @ ranks) / denom
            corr[a:b] = block.cpu()
            log(f"  [spearman] rows {b}/{N}")
    else:
        corr = ((ranks.T @ ranks) / denom).cpu()
    del ranks
    if device != 'cpu':
        torch.cuda.empty_cache()
    return corr.numpy().astype(np.float32)


def spatial_foreground_matrix(spatial_maps, foreground_quantile):
    N_img, N_neu, N_cell = spatial_maps.shape
    X = spatial_maps.transpose(1, 0, 2).reshape(N_neu, N_img * N_cell)
    thr = np.quantile(X, foreground_quantile, axis=1, keepdims=True)
    live = (X >= thr).any(axis=0)
    Xf = X[:, live]
    kept = int(live.sum())
    log(f"  [spatial] {N_img} imgs x {N_cell} cells = {N_img*N_cell} cells; "
        f"foreground_quantile={foreground_quantile} -> kept {kept} live "
        f"({100*kept/live.size:.1f}%)")
    if kept < 50:
        raise ValueError("Too few live cells; lower foreground_quantile.")
    return np.ascontiguousarray(Xf.T)


def build_graph_from_corr(corr, threshold):
    corr = corr.copy()
    np.fill_diagonal(corr, 0.0)
    N = corr.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    w = corr[iu, ju]
    keep = w > threshold
    log(f"  [graph] {int(keep.sum())} edges pass threshold {threshold} "
        f"(of {w.size} pairs)")
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(),
                                  w[keep].astype(float).tolist()))
    G.remove_nodes_from(list(nx.isolates(G)))
    log(f"  [graph] after removing isolates: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges")
    return G


# ======================================================================
# STAGE 1 -- BUILD (your notebook; external imports used as-is)
# ======================================================================

def _spatial_builder_class():
    from my_coactivation_graph import CoactivationGraphBuilder

    class SpatialCoactivationGraphBuilder(CoactivationGraphBuilder):
        def __init__(self, spatial_maps, foreground_quantile=0.8):
            X = np.asarray(spatial_maps, dtype=np.float32)
            if X.ndim != 3:
                raise ValueError(f"expected (N_images, N_neurons, N_cells), got {X.shape}")
            self.spatial_maps = X
            self.foreground_quantile = foreground_quantile
            super().__init__(X.mean(axis=2))

        def prepare_activation_matrix(self):
            N_img, N_neu, N_cell = self.spatial_maps.shape
            X = self.spatial_maps.transpose(1, 0, 2).reshape(N_neu, N_img * N_cell)
            thr = np.quantile(X, self.foreground_quantile, axis=1, keepdims=True)
            live = (X >= thr).any(axis=0)
            Xf = X[:, live]
            kept = int(live.sum())
            log(f"  [spatial substrate] kept {kept} live cells "
                f"({100*kept/live.size:.1f}%)")
            if kept < 50:
                raise ValueError("Too few live cells; lower foreground_quantile.")
            self.pooled_matrix = Xf
            return self.pooled_matrix, N_neu, kept

    return SpatialCoactivationGraphBuilder


def build_graphs_and_npz(cfg):
    import torch
    import torch.nn.functional as F
    import timm
    from torch.utils.data import DataLoader, Subset
    from my_coactivation_graph import CoactivationGraphBuilder
    from MnMs2DatasetConcpets import MnMsDatasetLAX

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    banner("STAGE 1: BUILD")
    log(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if DEVICE == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)} | "
            f"mem {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        log("WARNING: running on CPU -- this will be slow.")

    IMAGE_SIZE = cfg.image_size
    TOP_K_PERCENT = cfg.top_k_percent
    LAYER = cfg.layer
    DERIVED_CONCEPT_NAMES = list(cfg.concepts)
    log(f"NCA over {len(DERIVED_CONCEPT_NAMES)} concepts: {DERIVED_CONCEPT_NAMES}")

    log("loading DenseNet-161 + weights ...")
    model = timm.create_model('densenet161', pretrained=False, num_classes=8)
    ckpt = torch.load(cfg.model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE).eval()
    log(f"model on {next(model.parameters()).device}")

    spatial_activations = {}

    def spatial_hook(module, inp, out):
        spatial_activations['act'] = out.detach()

    dict(model.named_modules())[LAYER].register_forward_hook(spatial_hook)

    log("loading MnM2 LAX dataset (full image, get_original_concepts=True) ...")
    full_dataset = MnMsDatasetLAX(cfg.mnm2_root, max_images=None,
                                  get_original_concepts=True)
    la_indices = [idx for idx, rec in enumerate(full_dataset.image_slices)
                  if '_LA_' in rec[4]]
    balanced_dataset = Subset(full_dataset, la_indices)
    log(f"{len(la_indices)} LA slices selected")

    def lax_collate_fn(batch):
        imgs, gt_slices, lbls, diseases, patients, slices, files, views, concepts = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        lbls = torch.as_tensor(lbls, dtype=torch.long)
        slices = torch.as_tensor(slices, dtype=torch.long)
        return (imgs, list(gt_slices), lbls, list(diseases), list(patients),
                slices, list(files), list(views), list(concepts))

    loader = DataLoader(balanced_dataset, batch_size=min(64, len(balanced_dataset)),
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=(DEVICE == 'cuda'), collate_fn=lax_collate_fn)
    n_batches = len(loader)

    work_device = torch.device(DEVICE)
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    quantile_q = 1.0 - TOP_K_PERCENT
    pooled_activations, spatial_maps, sample_images, sample_ids, concept_masks = [], [], [], [], []
    association_sum = association_count = None
    n_processed = n_skipped = 0

    log(f"forward pass over {n_batches} batches ...")
    for bi, batch in enumerate(loader):
        imgs, gt_slices, lbls, diseases, patients, slices, files, views, concepts_batch = batch
        imgs = imgs.to(work_device, non_blocking=(DEVICE == 'cuda'))
        with torch.no_grad():
            _ = model(imgs)
        batch_act_map = spatial_activations['act']
        B, N_neurons, gh, gw = batch_act_map.shape
        if association_sum is None:
            association_sum = torch.zeros((N_neurons, len(DERIVED_CONCEPT_NAMES)),
                                          dtype=torch.float32, device=work_device)
            association_count = torch.zeros((N_neurons, len(DERIVED_CONCEPT_NAMES)),
                                            dtype=torch.float32, device=work_device)
        pooled_batch = batch_act_map.mean(dim=(2, 3)).detach().cpu().numpy()
        spatial_batch = batch_act_map.reshape(B, N_neurons, gh * gw).detach().cpu().numpy()

        for i in range(B):
            if views[i] != 'LA':
                continue
            pooled_activations.append(pooled_batch[i])
            spatial_maps.append(spatial_batch[i])
            sample_images.append(imgs[i].detach().cpu().numpy())
            sample_ids.append(str(files[i]))
            sample_concepts = concepts_batch[i] if concepts_batch[i] is not None else {}
            concept_mask = np.zeros((len(DERIVED_CONCEPT_NAMES), IMAGE_SIZE, IMAGE_SIZE),
                                    dtype=np.uint8)
            names_present, mask_tensors = [], []
            for cname in DERIVED_CONCEPT_NAMES:
                mask_raw = sample_concepts.get(cname, None)
                if mask_raw is None:
                    continue
                mt = (mask_raw if torch.is_tensor(mask_raw)
                      else torch.as_tensor(mask_raw, dtype=torch.float32))
                mt = mt.to(work_device, dtype=torch.float32)
                if tuple(mt.shape[-2:]) != target_size:
                    mt = F.interpolate(mt.unsqueeze(0).unsqueeze(0), size=target_size,
                                       mode='nearest').squeeze(0).squeeze(0)
                mb = mt > 0.5
                if mb.sum() == 0:
                    continue
                names_present.append(cname)
                mask_tensors.append(mb)
                concept_mask[DERIVED_CONCEPT_NAMES.index(cname)] = mb.detach().cpu().numpy().astype(np.uint8)
            concept_masks.append(concept_mask)
            if not mask_tensors:
                n_skipped += 1
                n_processed += 1
                continue
            act_map = batch_act_map[i]
            up = F.interpolate(act_map.unsqueeze(1), size=target_size,
                               mode='bilinear', align_corners=False).squeeze(1)
            thr = torch.quantile(up.flatten(1), q=quantile_q, dim=1, keepdim=True)
            patches = up >= thr.view(-1, 1, 1)
            masks_stack = torch.stack(mask_tensors, dim=0)
            inter = (patches.unsqueeze(1) & masks_stack.unsqueeze(0)).sum(dim=(2, 3)).float()
            union = (patches.unsqueeze(1) | masks_stack.unsqueeze(0)).sum(dim=(2, 3)).float()
            ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))
            for j, cname in enumerate(names_present):
                c_idx = DERIVED_CONCEPT_NAMES.index(cname)
                association_sum[:, c_idx] += ious[:, j]
                association_count[:, c_idx] += 1.0
            n_processed += 1
        log(f"  [forward] batch {bi+1}/{n_batches} | images processed {n_processed}")

    association_matrix = torch.where(association_count > 0,
                                     association_sum / association_count,
                                     torch.zeros_like(association_sum))
    association_matrix_np = association_matrix.detach().cpu().numpy()
    pooled_activations = np.asarray(pooled_activations)
    spatial_maps = np.asarray(spatial_maps, dtype=np.float32)
    sample_images = np.asarray(sample_images, dtype=np.float32)
    concept_masks = np.asarray(concept_masks, dtype=np.uint8)
    assert np.allclose(pooled_activations, spatial_maps.mean(axis=2), atol=1e-5)
    log(f"forward pass done: {n_processed} imgs ({n_skipped} w/o concepts) | "
        f"pooled {pooled_activations.shape} | spatial {spatial_maps.shape}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    T = float(cfg.threshold)
    corr_spatial = np.empty((0, 0), dtype=np.float32)

    if cfg.use_gpu_spearman:
        banner("Spearman correlation on GPU")
        log("POOLED graph ...")
        corr_pooled = gpu_spearman_from_samples(pooled_activations, DEVICE,
                                                chunk_neurons=cfg.spearman_chunk)
        G_pooled = build_graph_from_corr(corr_pooled, T)
        nx.write_graphml(G_pooled, os.path.join(cfg.results_dir, pooled_graph_name(cfg)))
        log(f"saved {pooled_graph_name(cfg)}")

        if cfg.build_spatial_graph:
            log("SPATIAL graph ...")
            Xf = spatial_foreground_matrix(spatial_maps, cfg.foreground_quantile)
            corr_spatial = gpu_spearman_from_samples(Xf, DEVICE,
                                                     chunk_neurons=cfg.spearman_chunk)
            G_spatial = build_graph_from_corr(corr_spatial, T)
            nx.write_graphml(G_spatial, os.path.join(cfg.results_dir, spatial_graph_name(cfg)))
            log(f"saved {spatial_graph_name(cfg)}")
        else:
            log("SPATIAL graph skipped")
    else:
        banner("Spearman correlation via my_coactivation_graph (CPU, as-is)")
        SpatialBuilder = _spatial_builder_class()
        log("POOLED graph ...")
        pooled_builder = CoactivationGraphBuilder(pooled_activations)
        corr_pooled = pooled_builder.compute_spearman_correlation()
        np.fill_diagonal(corr_pooled, 0)
        G_pooled = pooled_builder.build_graph(threshold=T)
        G_pooled.remove_nodes_from(list(nx.isolates(G_pooled)))
        pooled_builder.save_graph(G_pooled, cfg.results_dir + '/', pooled_graph_name(cfg))
        log(f"saved {pooled_graph_name(cfg)}")

        if cfg.build_spatial_graph:
            log("SPATIAL graph ...")
            spatial_builder = SpatialBuilder(spatial_maps, foreground_quantile=cfg.foreground_quantile)
            corr_spatial = spatial_builder.compute_spearman_correlation()
            np.fill_diagonal(corr_spatial, 0)
            spatial_builder.print_correlation_stats()
            G_spatial = spatial_builder.build_graph(threshold=T)
            G_spatial.remove_nodes_from(list(nx.isolates(G_spatial)))
            spatial_builder.save_graph(G_spatial, cfg.results_dir + '/', spatial_graph_name(cfg))
            log(f"saved {spatial_graph_name(cfg)}")
        else:
            log("SPATIAL graph skipped")

    payload = dict(
        association_matrix=association_matrix_np,
        concept_names=DERIVED_CONCEPT_NAMES,
        correlation_matrix=corr_pooled,
        correlation_matrix_spatial=corr_spatial,
    )
    if cfg.save_spatial_maps:
        payload['spatial_maps'] = spatial_maps
        payload['spatial_shape'] = np.asarray([gh, gw], dtype=np.int64)
    if cfg.save_visual_evidence_inputs:
        payload['sample_images'] = sample_images
        payload['sample_ids'] = np.asarray(sample_ids, dtype=object)
        payload['concept_masks'] = concept_masks
    np.savez_compressed(os.path.join(cfg.results_dir, npz_name(cfg)), **payload)
    log(f"saved {npz_name(cfg)}")
    log("BUILD complete.")


def pooled_graph_name(cfg):
    nc = len(cfg.concepts)
    return f"coactivation_{nc}_dice0.06_{cfg.image_tag}_{float(cfg.threshold)}_B{cfg.block}.graphml"


def spatial_graph_name(cfg):
    nc = len(cfg.concepts)
    return f"coactivation_{nc}_dice0.06_{cfg.image_tag}_{float(cfg.threshold)}_B{cfg.block}_SPATIAL.graphml"


def npz_name(cfg):
    nc = len(cfg.concepts)
    return f"{nc}_{cfg.image_tag}_dice0.06_B{cfg.block}_concept_associations_SPATIAL.npz"


# ======================================================================
# STAGE 2 -- ANALYSIS
# ======================================================================

def _node_to_index(node):
    if isinstance(node, (int, np.integer)):
        return int(node)
    m = re.findall(r'\d+', str(node))
    if not m:
        raise ValueError(f"cannot parse neuron index from node id {node!r}")
    return int(m[0])


def load_graph(path, threshold=None, label=None):
    log(f"  reading {os.path.basename(path)} ...")
    G = nx.Graph(nx.read_graphml(path))
    for n in G.nodes():
        G.nodes[n]['nidx'] = _node_to_index(n)
    for u, v, d in G.edges(data=True):
        try:
            d['weight'] = float(d.get('weight', 1.0))
        except (TypeError, ValueError):
            d['weight'] = 1.0
    if threshold is not None:
        threshold=0.3 if label =="pooled" else 0.15
        builder = CoactivationGraphBuilder(None)
        G = builder.apply_threshold(G, threshold)
    return G


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    assoc = np.asarray(d['association_matrix'], dtype=np.float64)
    names = [str(x) for x in d['concept_names']]
    spatial_maps = np.asarray(d['spatial_maps'], dtype=np.float32) if 'spatial_maps' in d else None
    sample_images = np.asarray(d['sample_images'], dtype=np.float32) if 'sample_images' in d else None
    sample_ids = [str(x) for x in d['sample_ids']] if 'sample_ids' in d else None
    concept_masks = np.asarray(d['concept_masks'], dtype=np.uint8) if 'concept_masks' in d else None
    spatial_shape = tuple(int(x) for x in d['spatial_shape']) if 'spatial_shape' in d else None
    extras = dict(sample_images=sample_images, sample_ids=sample_ids,
                  concept_masks=concept_masks, spatial_shape=spatial_shape)
    return assoc, names, spatial_maps, extras


def build_label_arrays(G, assoc):
    nodes = list(G.nodes())
    nidx = np.array([G.nodes[n]['nidx'] for n in nodes], dtype=int)
    rows = assoc[nidx]
    row_sum = rows.sum(axis=1, keepdims=True)
    has_concept = (row_sum[:, 0] > 0)
    C = rows.shape[1]
    profiles = np.where(row_sum > 0, rows / np.where(row_sum > 0, row_sum, 1.0), 1.0 / C)
    hard = rows.argmax(axis=1)
    return nodes, nidx, hard, profiles, has_concept


def edge_arrays(G, nodes):
    pos = {n: i for i, n in enumerate(nodes)}
    eu, ev, ew = [], [], []
    for u, v, d in G.edges(data=True):
        eu.append(pos[u]); ev.append(pos[v]); ew.append(d['weight'])
    return (np.asarray(eu, dtype=np.int64), np.asarray(ev, dtype=np.int64),
            np.asarray(ew, dtype=np.float64))


def jsd_distance_edges(P_u, P_v):
    m = 0.5 * (P_u + P_v)

    def _kl(p, q):
        mask = p > 0
        out = np.zeros(p.shape, dtype=np.float64)
        out[mask] = p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))
        return out.sum(axis=1)

    div = np.clip(0.5 * _kl(P_u, m) + 0.5 * _kl(P_v, m), 0.0, 1.0)
    return np.sqrt(div)


def edge_agreement(hard, profiles, eu, ev, mode, tau_jsd):
    if mode == 'hard':
        return hard[eu] == hard[ev]
    if mode == 'jsd':
        return jsd_distance_edges(profiles[eu], profiles[ev]) < tau_jsd
    raise ValueError(mode)


def node_homophily_scores(agree, eu, ev, n_nodes):
    deg = np.bincount(np.concatenate([eu, ev]), minlength=n_nodes).astype(np.float64)
    same = np.bincount(np.concatenate([eu[agree], ev[agree]]),
                       minlength=n_nodes).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        score = np.where(deg > 0, same / deg, np.nan)
    return score, deg


def permutation_null(hard, profiles, eu, ev, mode, tau_jsd, n_perm, seed):
    rng = np.random.default_rng(seed)
    n_nodes = hard.shape[0]
    agree_obs = edge_agreement(hard, profiles, eu, ev, mode, tau_jsd)
    eh_obs = float(agree_obs.mean())
    nh_score, deg = node_homophily_scores(agree_obs, eu, ev, n_nodes)
    nh_obs = float(np.nanmean(nh_score))
    eh_null = np.empty(n_perm); nh_null = np.empty(n_perm)
    idx = np.arange(n_nodes)
    step = max(1, n_perm // 10)
    log(f"    [null · {mode}] {n_perm} label permutations ...")
    for k in range(n_perm):
        perm = rng.permutation(idx)
        if mode == 'hard':
            ag = hard[perm][eu] == hard[perm][ev]
        else:
            ag = jsd_distance_edges(profiles[perm][eu], profiles[perm][ev]) < tau_jsd
        eh_null[k] = ag.mean()
        same = np.bincount(np.concatenate([eu[ag], ev[ag]]),
                           minlength=n_nodes).astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            nh_null[k] = np.nanmean(np.where(deg > 0, same / deg, np.nan))
        if (k + 1) % step == 0:
            log(f"    [null · {mode}] {k+1}/{n_perm}")

    def _summ(obs, null):
        mu, sd = float(null.mean()), float(null.std(ddof=1) + 1e-12)
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        return dict(observed=round(obs, 4), null_mean=round(mu, 4),
                    ratio=round(obs / mu, 2) if mu > 0 else float('inf'),
                    z=round((obs - mu) / sd, 1), p_value=round(float(p), 5))

    return {'edge': _summ(eh_obs, eh_null), 'node': _summ(nh_obs, nh_null)}


# ---------- substructures (now carry member_idx for aggregate metrics) ----------

def substructures_from_edges(nodes, eu, ev, keep_mask, hard, min_size=3):
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    kept = np.where(keep_mask)[0]
    H.add_edges_from(zip(eu[kept].tolist(), ev[kept].tolist()))
    subs = []
    for comp in nx.connected_components(H):
        comp = [c for c in comp if H.degree(c) > 0]
        if len(comp) < min_size:
            continue
        labs = hard[list(comp)]
        vals, counts = np.unique(labs, return_counts=True)
        subs.append(dict(members=[nodes[c] for c in comp],
                         member_idx=[int(c) for c in comp], size=len(comp),
                         concept=int(vals[counts.argmax()]),
                         purity=round(float(counts.max() / counts.sum()), 3)))
    subs.sort(key=lambda s: s['size'], reverse=True)
    return subs


def node_substructures(nodes, eu, ev, agree, hard, node_score, node_thresh, min_size=3):
    """Mode-specific agreement edges restricted to high-homophily core nodes."""
    core = node_score >= node_thresh
    keep = agree & core[eu] & core[ev]
    return substructures_from_edges(nodes, eu, ev, keep, hard, min_size=min_size)


def combined_substructures(nodes, eu, ev, agree, hard, node_score,
                           node_thresh, min_size=3):
    """Mode-specific edge agreement restricted to high-homophily core nodes."""
    core = node_score >= node_thresh
    keep = agree & core[eu] & core[ev]
    return substructures_from_edges(nodes, eu, ev, keep, hard, min_size=min_size)


def clustering_purity(comm_labels, concept_labels):
    total, hit = len(concept_labels), 0
    for c in np.unique(comm_labels):
        mask = comm_labels == c
        if mask.sum():
            _, counts = np.unique(concept_labels[mask], return_counts=True)
            hit += counts.max()
    return hit / total


def substructure_report(subs, hard, n_nodes, concept_names, top=8):
    """Substructure-wise + aggregate-as-clustering metrics.
    Returns (agg dict, top table, full table)."""
    if not subs:
        agg = dict(n=0, coverage=0.0, purity=float('nan'), nmi=float('nan'),
                   ari=float('nan'), size_min=0, size_med=0, size_max=0, size_mean=0.0)
        return agg, [], []
    assign = np.full(n_nodes, -1, dtype=int)
    for sid, s in enumerate(subs):
        for idx in s['member_idx']:
            assign[idx] = sid
    covered = assign >= 0
    sl, cl = assign[covered], hard[covered]
    sizes = np.array([s['size'] for s in subs])
    agg = dict(
        n=len(subs), coverage=round(float(covered.sum() / n_nodes), 3),
        purity=round(float(clustering_purity(sl, cl)), 3),
        nmi=round(float(normalized_mutual_info_score(cl, sl)), 3),
        ari=round(float(adjusted_rand_score(cl, sl)), 3),
        size_min=int(sizes.min()), size_med=int(np.median(sizes)),
        size_max=int(sizes.max()), size_mean=round(float(sizes.mean()), 1))
    table = []
    for sid, s in enumerate(subs):
        labs = hard[s['member_idx']]
        vals, counts = np.unique(labs, return_counts=True)
        probs = counts / counts.sum()
        ent = float(-(probs * np.log2(probs)).sum())
        table.append(dict(id=sid, size=s['size'],
                          dominant=concept_names[s['concept']],
                          purity=s['purity'], entropy=round(ent, 3),
                          composition={concept_names[int(v)]: int(c)
                                       for v, c in zip(vals, counts)}))
    table.sort(key=lambda r: r['size'], reverse=True)
    return agg, table[:top], table


def _infer_spatial_shape(spatial_maps, spatial_shape):
    if spatial_shape is not None:
        return spatial_shape
    n_cells = int(spatial_maps.shape[2])
    side = int(round(np.sqrt(n_cells)))
    if side * side != n_cells:
        raise ValueError("spatial_shape missing and flattened map is not square")
    return side, side


def _image_for_plot(sample):
    img = np.asarray(sample, dtype=np.float64)
    if img.ndim == 3:
        if img.shape[0] in (1, 3):
            img = img.mean(axis=0)
        else:
            img = img.mean(axis=-1)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    return np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _activation_area_for_plot(flat_map, spatial_shape, quantile):
    act = np.asarray(flat_map, dtype=np.float64).reshape(spatial_shape)
    return act >= np.quantile(act, quantile)


def _activation_heatmap_for_plot(flat_map, spatial_shape):
    act = np.asarray(flat_map, dtype=np.float64).reshape(spatial_shape)
    lo, hi = np.percentile(act, [1, 99])
    if hi <= lo:
        lo, hi = float(act.min()), float(act.max())
    return np.clip((act - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _draw_area_vs_gt(ax, base_img, activation_area, gt_mask, title, alpha):
    h, w = base_img.shape
    ax.imshow(base_img, cmap='gray')
    ax.imshow(activation_area.astype(float), cmap='Reds', alpha=alpha,
              interpolation='nearest', extent=(0, w, h, 0), vmin=0, vmax=1)
    if gt_mask is not None and gt_mask.any():
        ax.contour(gt_mask.astype(float), levels=[0.5], colors='lime',
                   linewidths=1.2)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _draw_heatmap_vs_gt(ax, base_img, activation_map, gt_mask, title, alpha,
                        show_gt=True):
    h, w = base_img.shape
    ax.imshow(base_img, cmap='gray')
    ax.imshow(activation_map, cmap='inferno', alpha=alpha, interpolation='bilinear',
              extent=(0, w, h, 0), vmin=0, vmax=1)
    if show_gt and gt_mask is not None and gt_mask.any():
        ax.contour(gt_mask.astype(float), levels=[0.5], colors='lime',
                   linewidths=1.2)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _draw_focus_overlay(ax, base_img, activation_map, title, focus_quantile,
                        background_alpha, blur_sigma):
    h, w = base_img.shape
    thr = np.quantile(activation_map, focus_quantile)
    mask = (activation_map >= thr).astype(float)
    try:
        from scipy.ndimage import gaussian_filter, zoom
        blurred = gaussian_filter(base_img, sigma=blur_sigma)
        alpha = zoom(mask, (h / mask.shape[0], w / mask.shape[1]), order=1)[:h, :w]
    except Exception:
        blurred = base_img
        sy = int(np.ceil(h / mask.shape[0]))
        sx = int(np.ceil(w / mask.shape[1]))
        alpha = np.repeat(np.repeat(mask, sy, axis=0), sx, axis=1)[:h, :w]
    alpha = np.clip(alpha, 0.0, 1.0)
    focus_img = (blurred * background_alpha) * (1.0 - alpha) + base_img * alpha
    ax.imshow(focus_img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _representative_neurons(maps, max_neurons):
    mean_map = maps.mean(axis=0)
    mean_centered = mean_map - mean_map.mean()
    mean_norm = np.linalg.norm(mean_centered)
    if mean_norm <= 1e-12:
        strength = maps.mean(axis=1)
        return np.argsort(strength)[::-1][:max_neurons]
    centered = maps - maps.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1) * mean_norm
    corr = np.full(maps.shape[0], -np.inf, dtype=np.float64)
    ok = denom > 1e-12
    corr[ok] = (centered[ok] * mean_centered).sum(axis=1) / denom[ok]
    return np.argsort(corr)[::-1][:max_neurons]


def plot_spatial_evidence(subs, nidx, spatial_maps, visual_inputs,
                          concept_names, tag, mode, kind, cfg):
    if not subs or spatial_maps is None:
        return
    sample_images = visual_inputs.get('sample_images')
    concept_masks = visual_inputs.get('concept_masks')
    if sample_images is None or concept_masks is None:
        log("  [visual] sample MRI images or GT concept masks missing; visual evidence skipped "
            "(set cfg.build=True once with cfg.save_visual_evidence_inputs=True)")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spatial_shape = _infer_spatial_shape(spatial_maps, visual_inputs.get('spatial_shape'))
    sample_ids = visual_inputs.get('sample_ids') or [f"img_{i}" for i in range(spatial_maps.shape[0])]
    out_dir = os.path.join(cfg.out_dir, 'visual_evidence', tag, mode, kind)
    os.makedirs(out_dir, exist_ok=True)

    n_subs = min(len(subs), cfg.visual_top_subs)
    for sid, sub in enumerate(subs[:n_subs]):
        member_idx = np.asarray(sub['member_idx'], dtype=np.int64)
        neuron_ids = nidx[member_idx]
        sub_maps_all = spatial_maps[:, neuron_ids, :]
        image_strength = sub_maps_all.mean(axis=(1, 2))
        image_ids = np.argsort(image_strength)[::-1][:cfg.visual_top_images]

        rep_source = sub_maps_all[image_ids].mean(axis=0)
        rep_local = _representative_neurons(rep_source, min(cfg.visual_top_neurons, len(neuron_ids)))
        rep_neurons = neuron_ids[rep_local]

        n_cols = 2 + len(rep_local)
        n_rows = len(image_ids)
        fig_w = max(7.0, 2.2 * n_cols)
        fig_h = max(3.0, 2.2 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        for r, img_id in enumerate(image_ids):
            base_img = _image_for_plot(sample_images[img_id])
            concept_idx = int(sub['concept'])
            gt_mask = concept_masks[img_id, concept_idx]
            mean_area = _activation_area_for_plot(sub_maps_all[img_id].mean(axis=0),
                                                  spatial_shape, cfg.visual_activation_quantile)

            ax = axes[r, 0]
            ax.imshow(base_img, cmap='gray')
            if gt_mask.any():
                ax.contour(gt_mask.astype(float), levels=[0.5], colors='lime',
                           linewidths=1.2)
            ax.set_title(str(sample_ids[img_id])[:36], fontsize=8)
            ax.axis('off')

            ax = axes[r, 1]
            _draw_area_vs_gt(ax, base_img, mean_area, gt_mask,
                             f"mean area vs {concept_names[concept_idx]}",
                             cfg.visual_overlay_alpha)

            for c, local_i in enumerate(rep_local, start=2):
                area = _activation_area_for_plot(sub_maps_all[img_id, local_i],
                                                 spatial_shape, cfg.visual_activation_quantile)
                ax = axes[r, c]
                _draw_area_vs_gt(ax, base_img, area, gt_mask,
                                 f"neuron {int(rep_neurons[c-2])}",
                                 cfg.visual_overlay_alpha)

        title = (f"{tag} {mode} {kind} S{sid} "
                 f"{concept_names[sub['concept']]} n={sub['size']} p={sub['purity']} "
                 f"dice={sub.get('spatial_dice', float('nan'))}")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        safe_name = f"S{sid:03d}_{concept_names[sub['concept']]}_n{sub['size']}.png"
        fig.savefig(os.path.join(out_dir, safe_name), dpi=cfg.visual_dpi, bbox_inches='tight')
        plt.close(fig)
    log(f"  [visual] saved {n_subs} evidence plots for {tag}/{mode}/{kind}")


def _need_visual_inputs(spatial_maps, visual_inputs):
    return (spatial_maps is not None and visual_inputs.get('sample_images') is not None
            and visual_inputs.get('concept_masks') is not None)


def _saved_substructures(concept_names, spatial_maps, cfg):
    base = os.path.join(cfg.out_dir, f"analysis_B{cfg.block}")
    json_path = base + "_substructures.json"
    csv_path = base + "_substructures_detail.csv"
    if not (os.path.exists(json_path) and os.path.exists(csv_path)):
        return None

    with open(json_path) as f:
        saved = json.load(f)
    detail = pd.read_csv(csv_path)
    concept_to_id = {c: i for i, c in enumerate(concept_names)}
    out = []
    for graph_rec in saved:
        tag = graph_rec['graph']
        if tag not in cfg.qualitative_graphs:
            continue
        for mode in cfg.qualitative_modes:
            local = graph_rec.get('local', {}).get(mode, {})
            for kind in cfg.qualitative_kinds:
                rows = detail[(detail.graph == tag) & (detail.labelling == mode)
                              & (detail.type == kind)].sort_values('sub_id')
                members = local.get('substructures', {}).get(kind, {}).get('members', [])
                for _, row in rows.head(cfg.qualitative_top_subs).iterrows():
                    sid = int(row.sub_id)
                    if sid >= len(members) or row.dominant not in concept_to_id:
                        continue
                    member_rec = members[sid]
                    raw_members = (member_rec.get('members', member_rec)
                                   if isinstance(member_rec, dict) else member_rec)
                    member_idx = [_node_to_index(n) for n in raw_members]
                    member_idx = [n for n in member_idx if 0 <= n < spatial_maps.shape[1]]
                    out.append(dict(tag=tag, mode=mode, kind=kind, sid=sid,
                                    member_idx=member_idx,
                                    concept=concept_to_id[row.dominant],
                                    size=int(row.size),
                                    purity=float(row.purity)))
    return out


def _representative_substructure_neurons(neuron_ids, spatial_maps, max_neurons):
    maps = spatial_maps[:, neuron_ids, :].mean(axis=0)
    local = _representative_neurons(maps, min(max_neurons, len(neuron_ids)))
    return [int(neuron_ids[i]) for i in local]


def plot_neuron_evidence(assoc, concept_names, spatial_maps, visual_inputs, cfg):
    if not _need_visual_inputs(spatial_maps, visual_inputs):
        log("  [qual] missing spatial maps/images/concept masks; neuron evidence skipped")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spatial_shape = _infer_spatial_shape(spatial_maps, visual_inputs.get('spatial_shape'))
    sample_images = visual_inputs['sample_images']
    concept_masks = visual_inputs['concept_masks']
    sample_ids = visual_inputs.get('sample_ids') or [f"img_{i}" for i in range(spatial_maps.shape[0])]
    max_neuron = spatial_maps.shape[1] - 1

    out_root = os.path.join(cfg.out_dir, 'substructure_neuron_evidence')
    substructures = _saved_substructures(concept_names, spatial_maps, cfg) or []
    jobs = []
    if cfg.qualitative_neuron_ids:
        jobs = [dict(neuron=int(n), concept=int(np.argmax(assoc[int(n)])),
                     group='manual', label='manual')
                for n in cfg.qualitative_neuron_ids if 0 <= int(n) <= max_neuron]
    else:
        for sub in substructures:
            reps = _representative_substructure_neurons(
                sub['member_idx'], spatial_maps, cfg.qualitative_top_neurons_per_substructure)
            label = f"{sub['tag']}_{sub['mode']}_{sub['kind']}_S{sub['sid']}_{concept_names[sub['concept']]}"
            jobs.extend(dict(neuron=n, concept=sub['concept'], group=label, label=label)
                        for n in reps)

    n_saved = 0
    for job in jobs:
        neuron = job['neuron']
        concept_idx = job['concept']
        scores = spatial_maps[:, neuron, :].mean(axis=1)
        image_ids = np.argsort(scores)[::-1][:cfg.qualitative_top_images]
        out_dir = os.path.join(out_root, job['group'])
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(len(image_ids), 3, figsize=(8.1, 2.7 * len(image_ids)),
                                 squeeze=False)
        for r, img_id in enumerate(image_ids):
            base_img = _image_for_plot(sample_images[img_id])
            gt_mask = concept_masks[img_id, concept_idx]
            heatmap = _activation_heatmap_for_plot(spatial_maps[img_id, neuron],
                                                   spatial_shape)
            axes[r, 0].imshow(base_img, cmap='gray')
            if gt_mask.any():
                axes[r, 0].contour(gt_mask.astype(float), levels=[0.5], colors='lime',
                                   linewidths=1.2)
            axes[r, 0].set_title(str(sample_ids[img_id])[:36], fontsize=8)
            axes[r, 0].axis('off')
            _draw_heatmap_vs_gt(axes[r, 1], base_img, heatmap, gt_mask,
                                f"neuron {neuron} vs {concept_names[concept_idx]}",
                                cfg.visual_overlay_alpha, show_gt=False)
            _draw_focus_overlay(axes[r, 2], base_img, heatmap,
                                "focus overlay", cfg.qualitative_focus_quantile,
                                cfg.qualitative_background_alpha,
                                cfg.qualitative_blur_sigma)
        fig.suptitle(f"{job['label']} | neuron {neuron} vs {concept_names[concept_idx]}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"neuron_{neuron}.png"),
                    dpi=cfg.visual_dpi, bbox_inches='tight')
        plt.close(fig)
        n_saved += 1
    log(f"  [qual] saved {n_saved} neuron evidence plots")


def plot_consensus_evidence(subs, nidx, spatial_maps, visual_inputs,
                            concept_names, tag, mode, kind, cfg):
    if not subs or not _need_visual_inputs(spatial_maps, visual_inputs):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spatial_shape = _infer_spatial_shape(spatial_maps, visual_inputs.get('spatial_shape'))
    sample_images = visual_inputs['sample_images']
    concept_masks = visual_inputs['concept_masks']
    sample_ids = visual_inputs.get('sample_ids') or [f"img_{i}" for i in range(spatial_maps.shape[0])]
    out_dir = os.path.join(cfg.out_dir, 'qualitative_consensus', tag, mode, kind)
    os.makedirs(out_dir, exist_ok=True)

    n_subs = min(len(subs), cfg.qualitative_top_subs)
    for sid, sub in enumerate(subs[:n_subs]):
        neuron_ids = nidx[np.asarray(sub['member_idx'], dtype=np.int64)]
        consensus = spatial_maps[:, neuron_ids, :].mean(axis=1)
        image_ids = np.argsort(consensus.mean(axis=1))[::-1][:cfg.qualitative_top_images]
        concept_idx = int(sub['concept'])

        fig, axes = plt.subplots(len(image_ids), 3, figsize=(8.1, 2.7 * len(image_ids)),
                                 squeeze=False)
        for r, img_id in enumerate(image_ids):
            base_img = _image_for_plot(sample_images[img_id])
            gt_mask = concept_masks[img_id, concept_idx]
            heatmap = _activation_heatmap_for_plot(consensus[img_id], spatial_shape)
            axes[r, 0].imshow(base_img, cmap='gray')
            if gt_mask.any():
                axes[r, 0].contour(gt_mask.astype(float), levels=[0.5], colors='lime',
                                   linewidths=1.2)
            axes[r, 0].set_title(str(sample_ids[img_id])[:36], fontsize=8)
            axes[r, 0].axis('off')
            _draw_heatmap_vs_gt(axes[r, 1], base_img, heatmap, gt_mask,
                                "substructure consensus", cfg.visual_overlay_alpha,
                                show_gt=False)
            _draw_focus_overlay(axes[r, 2], base_img, heatmap,
                                "focus overlay", cfg.qualitative_focus_quantile,
                                cfg.qualitative_background_alpha,
                                cfg.qualitative_blur_sigma)
        fig.suptitle(f"{tag} {mode} {kind} S{sid} {concept_names[concept_idx]} "
                     f"n={sub['size']} p={sub['purity']}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"consensus_S{sid}_{concept_names[concept_idx]}.png"),
                    dpi=cfg.visual_dpi, bbox_inches='tight')
        plt.close(fig)
    log(f"  [qual] saved {n_subs} consensus plots for {tag}/{mode}/{kind}")


def plot_saved_consensus_evidence(concept_names, spatial_maps, visual_inputs, cfg):
    if not _need_visual_inputs(spatial_maps, visual_inputs):
        log("  [qual] missing spatial maps/images/concept masks; consensus evidence skipped")
        return True
    saved_subs = _saved_substructures(concept_names, spatial_maps, cfg)
    if saved_subs is None:
        log("  [qual] saved substructure tables missing; falling back to analysis")
        return False

    nidx = np.arange(spatial_maps.shape[1])
    groups = defaultdict(list)
    for sub in saved_subs:
        groups[(sub['tag'], sub['mode'], sub['kind'])].append(sub)
    for (tag, mode, kind), subs in groups.items():
        plot_consensus_evidence(subs, nidx, spatial_maps, visual_inputs,
                                concept_names, tag, mode, kind, cfg)
    return True


# ---------- on-graph substructure plotting ----------

def compute_layout(G, seed, iters):
    log(f"  [plot] spring layout for {G.number_of_nodes()} nodes "
        f"/ {G.number_of_edges()} edges ({iters} iters) ...")
    t0 = time.time()
    pos = nx.spring_layout(G, seed=seed, iterations=iters)
    log(f"  [plot] layout done in {time.time()-t0:.1f}s")
    return pos


def plot_substructures(nodes, pos, eu, ev, families, hard, concept_names,
                       tag, mode, out_dir, cfg):
    """Draw the graph for each substructure family, colouring each substructure
    a distinct colour and showing where it sits in the layout. Each family is
    saved as its OWN image file (not combined into one figure)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    os.makedirs(out_dir, exist_ok=True)
    posarr = np.array([pos[n] for n in nodes])
    cmap = plt.get_cmap('tab20')

    for fam, subs in families.items():
        rng = np.random.default_rng(cfg.seed)
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        # all nodes grey background
        ax.scatter(posarr[:, 0], posarr[:, 1], s=4, c='0.82',
                   linewidths=0, zorder=1)
        # faint background-edge sample for context
        if cfg.plot_background_edges and len(eu) > 0:
            ne = min(cfg.plot_background_edges, len(eu))
            sel = rng.choice(len(eu), ne, replace=False)
            segs = np.stack([posarr[eu[sel]], posarr[ev[sel]]], axis=1)
            ax.add_collection(LineCollection(segs, colors='0.88',
                                             linewidths=0.2, zorder=0))

        out_path = os.path.join(
            out_dir, f"substructures_B{cfg.block}_{tag}_{mode}_{fam}.png")

        if not subs:
            ax.set_title(f"{tag} · {fam} · {mode}\n(no substructures)", fontsize=12)
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(out_path, dpi=cfg.plot_dpi, bbox_inches='tight')
            plt.close(fig)
            log(f"  [plot] saved {os.path.basename(out_path)}")
            continue

        # map node -> drawn-substructure id (only the largest plot_max_subs)
        n_draw = min(len(subs), cfg.plot_max_subs)
        sub_of = np.full(len(nodes), -1, dtype=int)
        for sid in range(n_draw):
            for idx in subs[sid]['member_idx']:
                sub_of[idx] = sid

        # intra-substructure edges, coloured by substructure (capped)
        intra = (sub_of[eu] >= 0) & (sub_of[eu] == sub_of[ev])
        ie = np.where(intra)[0]
        if ie.size > cfg.plot_max_edges:
            ie = rng.choice(ie, cfg.plot_max_edges, replace=False)
        if ie.size:
            segs = np.stack([posarr[eu[ie]], posarr[ev[ie]]], axis=1)
            ecols = [cmap(sub_of[eu[e]] % 20) for e in ie]
            ax.add_collection(LineCollection(segs, colors=ecols,
                                             linewidths=0.35, alpha=0.5, zorder=2))

        # substructure nodes on top, one colour each + legend for the largest
        handles = []
        for sid in range(n_draw):
            idx = subs[sid]['member_idx']
            col = cmap(sid % 20)
            ax.scatter(posarr[idx, 0], posarr[idx, 1], s=16, color=col,
                       linewidths=0, zorder=3)
            if sid < cfg.plot_legend_top:
                s = subs[sid]
                handles.append(plt.Line2D(
                    [0], [0], marker='o', linestyle='', color=col,
                    label=f"S{sid} {concept_names[s['concept']]} "
                          f"n={s['size']} p={s['purity']}"))
        ax.set_title(f"{tag} · {fam} · {mode}  "
                     f"({len(subs)} subs, top {n_draw} shown)", fontsize=12)
        if handles:
            ax.legend(handles=handles, fontsize=8, loc='upper right',
                      framealpha=0.85, markerscale=1.2)
        ax.axis('off')

        fig.tight_layout()
        fig.savefig(out_path, dpi=cfg.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        log(f"  [plot] saved {os.path.basename(out_path)}")


# ---------- global: communities, pagerank ----------

def community_analysis(G, nodes, hard, profiles, concept_names, resolution, seed):
    n_nodes = len(nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    log("  [global] clamping negative weights, running Louvain ...")
    Gp = G.copy()
    for u, v, d in Gp.edges(data=True):
        d['weight'] = max(float(d.get('weight', 1.0)), 0.0)
    if louvain_communities is None:
        raise RuntimeError("networkx too old for louvain_communities; upgrade networkx.")
    comms = louvain_communities(Gp, weight='weight', resolution=resolution, seed=seed)
    log(f"  [global] {len(comms)} raw communities | computing PageRank ...")
    comm_label = np.full(n_nodes, -1, dtype=int)
    for ci, com in enumerate(comms):
        for n in com:
            comm_label[pos[n]] = ci
    pr = nx.pagerank(Gp, weight='weight')
    pr_arr = np.array([pr[n] for n in nodes])
    rows = []
    for ci, com in enumerate(comms):
        idx = [pos[n] for n in com]
        if len(idx) < 2:
            continue
        sub_pr = pr_arr[idx]
        labs = hard[idx]
        vals, counts = np.unique(labs, return_counts=True)
        rows.append(dict(community=ci, size=len(idx),
                         influential_neuron=nodes[idx[int(np.argmax(sub_pr))]],
                         influential_pagerank=round(float(sub_pr.max()), 5),
                         dominant_concept=concept_names[int(vals[counts.argmax()])],
                         purity=round(float(counts.max() / counts.sum()), 3),
                         profile=np.round(profiles[idx].mean(axis=0), 3).tolist()))
    rows.sort(key=lambda r: r['size'], reverse=True)
    metrics = dict(
        n_communities=len([c for c in comms if len(c) >= 2]),
        modularity=round(nx.algorithms.community.modularity(Gp, comms, weight='weight'), 4),
        purity=round(float(clustering_purity(comm_label, hard)), 4),
        nmi=round(float(normalized_mutual_info_score(hard, comm_label)), 4),
        ari=round(float(adjusted_rand_score(hard, comm_label)), 4))
    return metrics, rows


def analyse_graph(graph_path, assoc, concept_names, spatial_maps, visual_inputs, cfg, tag):
    banner(f"{tag.upper()} GRAPH  ({os.path.basename(graph_path)})")
    G = load_graph(graph_path, threshold=0.30, label=tag)
    nodes, nidx, hard, profiles, has_concept = build_label_arrays(G, assoc)
    eu, ev, ew = edge_arrays(G, nodes)
    n_nodes, n_edges = len(nodes), len(eu)
    log(f"  nodes={n_nodes} edges={n_edges} "
        f"neurons-with-concept={int(has_concept.sum())}/{n_nodes}")
    out = dict(tag=tag, n_nodes=n_nodes, n_edges=n_edges)

    gmet, grows = community_analysis(G, nodes, hard, profiles, concept_names,
                                     cfg.resolution, cfg.seed)
    out['global'] = gmet; out['communities'] = grows
    print("\n[global] communities / pagerank / clustering vs hard-argmax")
    print(pd.DataFrame([gmet]).to_string(index=False))
    print("\n  top communities (by size):")
    print(pd.DataFrame(grows)[['community', 'size', 'dominant_concept', 'purity',
                               'influential_neuron', 'influential_pagerank']]
          .head(8).to_string(index=False))

    # concept assortativity (hard labels; mode-independent) -- computed once
    for i, n in enumerate(nodes):
        G.nodes[n]['concept'] = int(hard[i])
    assort = round(float(nx.attribute_assortativity_coefficient(G, 'concept')), 4)

    layout = None  # computed lazily, reused across families and modes
    out['local'] = {}
    for mode in ['hard', 'jsd']:
        log(f"  [local · {mode}] homophily + substructures ...")
        agree = edge_agreement(hard, profiles, eu, ev, mode, cfg.jsd_tau)
        nh_score, deg = node_homophily_scores(agree, eu, ev, n_nodes)
        null = permutation_null(hard, profiles, eu, ev, mode, cfg.jsd_tau,
                                cfg.n_perm, cfg.seed)
        node_summary = dict(mean_node_homophily=round(float(np.nanmean(nh_score)), 4),
                            core_nodes=int(np.nansum(nh_score >= cfg.node_thresh)))

        edge_subs = substructures_from_edges(nodes, eu, ev, agree, hard, min_size=cfg.min_size)
        node_subs = node_substructures(nodes, eu, ev, agree, hard, nh_score,
                                       cfg.node_thresh, min_size=cfg.min_size)
        comb_subs = combined_substructures(nodes, eu, ev, agree, hard, nh_score,
                                           cfg.node_thresh, min_size=cfg.min_size)

        reports = {}
        for kind, subs in [('edge', edge_subs), ('node', node_subs), ('combined', comb_subs)]:
            agg, top_tbl, full_tbl = substructure_report(subs, hard, n_nodes,
                                                         concept_names, top=8)
            reports[kind] = dict(agg=agg, table=full_tbl, _subs=subs)
            if (cfg.qualitative and tag in cfg.qualitative_graphs
                    and mode in cfg.qualitative_modes
                    and kind in cfg.qualitative_kinds):
                plot_consensus_evidence(subs, nidx, spatial_maps, visual_inputs,
                                        concept_names, tag, mode, kind, cfg)

        out['local'][mode] = dict(
            null=null, assortativity=assort, node=node_summary, substructures=reports)

        print(f"\n[local · {mode}] homophily vs label-permutation null")
        print(pd.DataFrame([null['edge'], null['node']], index=['edge', 'node']).to_string())
        print(f"  concept assortativity = {assort}")
        print(f"  mean node homophily   = {node_summary['mean_node_homophily']} "
              f"(core ≥{cfg.node_thresh}: {node_summary['core_nodes']})")
        for kind in ['edge', 'node', 'combined']:
            agg = reports[kind]['agg']
            print(f"\n  -- {kind} substructures (as a clustering): "
                  f"n={agg['n']} coverage={agg['coverage']} purity={agg['purity']} "
                  f"nmi={agg['nmi']} ari={agg['ari']} "
                  f"sizes[min/med/max]={agg['size_min']}/{agg['size_med']}/{agg['size_max']}")
            if reports[kind]['table']:
                tdf = pd.DataFrame([{k: v for k, v in t.items() if k != 'composition'}
                                    for t in reports[kind]['table'][:8]])
                print(tdf.to_string(index=False))

        # ---- draw the substructures on the graph ----
        if cfg.plot and mode in cfg.plot_modes:
            if layout is None:
                layout = compute_layout(G, cfg.seed, cfg.plot_layout_iters)
            fams = {'edge': edge_subs, 'node': node_subs, 'combined': comb_subs}
            plot_substructures(nodes, layout, eu, ev, fams, hard, concept_names,
                               tag, mode, cfg.out_dir, cfg)
    return out


def save_outputs(results, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.join(cfg.out_dir, f"analysis_B{cfg.block}")

    flat = []
    for r in results:
        for mode, rec in r['local'].items():
            row = dict(graph=r['tag'], labelling=mode, n_nodes=r['n_nodes'],
                       n_edges=r['n_edges'], assortativity=rec['assortativity'],
                       mean_node_homophily=rec['node']['mean_node_homophily'],
                       core_nodes=rec['node']['core_nodes'])
            for lvl in ['edge', 'node']:
                for k, v in rec['null'][lvl].items():
                    row[f"{lvl}hom_{k}"] = v
            for kind in ['edge', 'node', 'combined']:
                agg = rec['substructures'][kind]['agg']
                row.update({f"{kind}_n": agg['n'], f"{kind}_coverage": agg['coverage'],
                            f"{kind}_purity": agg['purity'], f"{kind}_nmi": agg['nmi'],
                            f"{kind}_ari": agg['ari'], f"{kind}_size_max": agg['size_max']})
            flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv(base + "_homophily_summary.csv", index=False)
    pd.DataFrame([{**{'graph': r['tag']}, **r['global']} for r in results]
                 ).to_csv(base + "_global_summary.csv", index=False)

    detail = []
    for r in results:
        for mode, rec in r['local'].items():
            for kind in ['edge', 'node', 'combined']:
                for t in rec['substructures'][kind]['table']:
                    detail.append(dict(graph=r['tag'], labelling=mode, type=kind,
                                       sub_id=t['id'], size=t['size'],
                                       dominant=t['dominant'], purity=t['purity'],
                                       entropy=t['entropy'],
                                       composition=json.dumps(t['composition'])))
    pd.DataFrame(detail).to_csv(base + "_substructures_detail.csv", index=False)

    sub_metrics = []
    for r in results:
        for mode, rec in r['local'].items():
            for kind in ['edge', 'node', 'combined']:
                agg = rec['substructures'][kind]['agg']
                sub_metrics.append(dict(graph=r['tag'], labelling=mode, type=kind, **agg))
    sub_df = pd.DataFrame(sub_metrics)
    sub_df.to_csv(base + "_substructure_metrics.csv", index=False)

    def _strip(rec):
        return {k: v for k, v in rec.items() if k != 'substructures'}
    serial = []
    for r in results:
        entry = dict(graph=r['tag'], global_=r['global'],
                     communities=r['communities'][:20], local={})
        for mode, rec in r['local'].items():
            entry['local'][mode] = dict(
                summary=_strip(rec),
                substructures={kind: dict(
                    agg=rec['substructures'][kind]['agg'],
                    members=[s['members'] for s in rec['substructures'][kind]['_subs'][:50]])
                    for kind in ['edge', 'node', 'combined']})
        serial.append(entry)
    with open(base + "_substructures.json", "w") as f:
        json.dump(serial, f, indent=2, default=str)

    log(f"saved: {base}_homophily_summary.csv / _global_summary.csv / "
        f"_substructure_metrics.csv / _substructures_detail.csv / _substructures.json")
    print("\n==== HOMOPHILY SUMMARY (all graphs × labelling rules) ====")
    print(df.to_string(index=False))
    print("\n==== SUBSTRUCTURE-WISE METRICS (edge / node / combined) ====")
    print(sub_df.to_string(index=False))


# ======================================================================
# MAIN  -- edit configuration here
# ======================================================================

class Cfg:
    pass


def main():
    cfg = Cfg()

    # ---- what to run ----
    cfg.build = True             # enriched B1 NPZ already has qualitative tensors
    cfg.build_spatial_graph = False
    cfg.analyse_spatial_graph = False
    cfg.save_spatial_maps = True
    cfg.save_visual_evidence_inputs = True
    cfg.use_gpu_spearman = False   # True = compute Spearman on the GPU (fast)
    cfg.spearman_chunk = 0        # 0 = no chunking; set e.g. 512 if you hit GPU OOM
    cfg.num_workers = 0        # sandbox-safe DataLoader; raise locally if wanted

    # ---- paths / identifiers ----
    cfg.results_dir = 'results_mnm2'
    cfg.block = 4
    cfg.threshold = 0.0           # FLOAT. filename uses str(0.0) -> "0.0"
    # all STAGE-2 analysis outputs (csv / json / png) go in their own subfolder
    # instead of being dumped into the cluttered results_dir
    cfg.out_dir = os.path.join('results_mnm2', f'analysis_3c_fullimg_B{cfg.block}')

    # ---- build-stage settings ----
    cfg.mnm2_root = '/media/kislay/New Volume/Turab/data/MnM2/'
    cfg.model_path = '/media/kislay/New Volume/Turab/CRAFT/models/best_densenet161_MnMs.pth'
    cfg.layer = f'features.denseblock{cfg.block}'
    cfg.top_k_percent = 0.06
    cfg.image_size = 224
    cfg.foreground_quantile = 0.8
    # ---- the three GT LA concepts on the FULL (uncropped) image ----
    cfg.image_tag = 'fullimg'     # filename token; keeps these distinct from the 'cropped' run
    cfg.concepts = ['LV', 'MYO', 'RV']

    # ---- analysis-stage settings ----
    cfg.jsd_tau = 0.35            # JSD distance below which an edge is concept-similar
    cfg.node_thresh = 0.6         # min node-homophily to be a "core" node
    cfg.resolution = 1.0          # Louvain resolution
    cfg.n_perm = 1000             # label permutations for the null
    cfg.min_size = 3              # min substructure size
    cfg.seed = 0


    # ---- post-hoc qualitative evidence ----
    cfg.qualitative = True
    cfg.qualitative_only = True
    cfg.qualitative_graphs = ['pooled', 'spatial']  # empty = all graphs; otherwise subset of tags
    cfg.qualitative_modes = ['jsd', 'hard']  # empty = all modes; otherwise subset of modes
    cfg.qualitative_kinds = ['node', 'combined', 'edge']  # empty = all kinds; otherwise subset of kinds
    cfg.qualitative_neuron_ids = []          # empty = representative neurons from saved substructures
    cfg.qualitative_top_neurons_per_substructure = 8
    cfg.qualitative_top_subs = 10
    cfg.qualitative_top_images = 6
    cfg.qualitative_focus_quantile = 0.90
    cfg.qualitative_background_alpha = 0.25
    cfg.qualitative_blur_sigma = 6.0
    cfg.visual_overlay_alpha = 0.55
    cfg.visual_dpi = 150

    # ---- substructure plotting ----
    cfg.plot = False              # master switch: draw substructures at all
    cfg.plot_hard = True          # include the hard-argmax labelling in the plots
    cfg.plot_jsd = True           # include the JSD labelling in the plots
    cfg.plot_modes = [m for m, on in [('hard', cfg.plot_hard), ('jsd', cfg.plot_jsd)] if on]
    cfg.plot_layout_iters = 100    # spring-layout iterations (more = slower, tidier)
    cfg.plot_max_subs = 20        # colour at most this many (largest) substructures
    cfg.plot_legend_top = 10      # legend entries for the largest substructures
    cfg.plot_max_edges = 6000     # cap intra-substructure edges drawn (speed)
    cfg.plot_background_edges = 2000  # faint context edges (0 to disable)
    cfg.plot_dpi = 150

    log(f"config: build={cfg.build} gpu_spearman={cfg.use_gpu_spearman} "
        f"results_dir={cfg.results_dir} block={cfg.block} threshold={cfg.threshold} "
        f"n_concepts={len(cfg.concepts)} image_tag={cfg.image_tag}")

    if cfg.build:
        build_graphs_and_npz(cfg)

    banner("STAGE 2: ANALYSIS")
    os.makedirs(cfg.out_dir, exist_ok=True)
    log(f"analysis outputs (csv / json / png) -> {cfg.out_dir}")
    pooled_path = os.path.join(cfg.results_dir, pooled_graph_name(cfg))
    npz_path = os.path.join(cfg.results_dir, npz_name(cfg))
    required_paths = [pooled_path, npz_path]
    spatial_path = os.path.join(cfg.results_dir, spatial_graph_name(cfg))
    if cfg.analyse_spatial_graph:
        required_paths.append(spatial_path)
    for pth in required_paths:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"{pth} not found -- set cfg.build=True or fix paths.")
    assoc, concept_names, spatial_maps, visual_inputs = load_npz(npz_path)
    log(f"NCA matrix {assoc.shape}, concepts={concept_names}")
    if spatial_maps is None:
        log("spatial activation maps not found in .npz; qualitative figures skipped "
            "(set cfg.build=True once to save them)")
    else:
        log(f"spatial activation maps {spatial_maps.shape}")
    if visual_inputs.get('sample_images') is None or visual_inputs.get('concept_masks') is None:
        log("sample MRI images or GT concept masks not found in .npz; qualitative figures skipped "
            "(set cfg.build=True once to save them)")
    if cfg.qualitative:
        plot_neuron_evidence(assoc, concept_names, spatial_maps, visual_inputs, cfg)
        if cfg.qualitative_only and plot_saved_consensus_evidence(
                concept_names, spatial_maps, visual_inputs, cfg):
            log("QUALITATIVE DONE.")
            return
    results = [analyse_graph(pooled_path, assoc, concept_names, spatial_maps, visual_inputs, cfg, 'pooled')]
    if cfg.analyse_spatial_graph:
        results.append(analyse_graph(spatial_path, assoc, concept_names,
                                     spatial_maps, visual_inputs, cfg, 'spatial'))
    save_outputs(results, cfg)
    log("ALL DONE.")


if __name__ == '__main__':
    main()
