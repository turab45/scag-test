# Independent SCAG Evaluation

This branch adds a patient-level, null-controlled evaluation without deleting or rewriting the original four analysis scripts.

## Why the new pipeline is necessary

The original scripts estimate the channel–concept matrix and activation-correlation graph from the same images, then use the same semantic profiles to construct and score local components. Perfect component purity/NMI/ARI can therefore occur by construction.

The new primary test is:

> Do activation-correlation edges estimated from one set of patients have greater anatomical-profile similarity on held-out patients than exact matched non-edges, with an aggregate whole-profile permutation test? Degree-preserving rewiring is reported separately as a secondary edge-similarity topology null.

### Prespecified primary contrast

Before inspecting the new outputs, the primary confirmatory contrast is fixed as:

- three original anatomical regions (LV/MYO/RV);
- full-image input, avoiding a ground-truth-mask-defined crop in the primary test;
- DenseNet Block 4, the final dense representation block;
- 5% positive-edge graph density;
- primary endpoint: held-out edge similarity minus exact degree/activity-stratum-matched non-edge similarity;
- aggregate patient-within-fold bootstrap interval and aggregate whole-profile permutation p-value.

Other blocks, cropped inputs, graph densities, and the nine derived regions are robustness/exploratory analyses. Their p-values must not be presented as additional independent confirmatory discoveries without multiplicity control.

The implementation uses:

- patient-level cross-fitting;
- patient-equal aggregation (patients with many slices are not overweighted);
- exact top-k spatial masks with abstention for constant/inactive channels;
- tie-correct Spearman correlation;
- fixed graph densities for comparable blocks;
- matched non-edges using train-side degree/activity strata;
- whole-profile permutation nulls;
- degree-preserving topology nulls;
- patient bootstrap intervals;
- descriptive global Louvain NMI/ARI and coupling–semantic dose-response results.

## Files

- `extract_scag_patient_data.py` — GPU extraction using the uploaded M&Ms-2 loaders and existing default paths.
- `scag_independent_alignment.py` — data-independent cross-fitted analysis and simulation mode.
- `scag_dataset_utils.py` — shared phase-matched GT resolver.
- `run_scag_gpu.sh` — reproducible smoke, primary, extended, and complete runs.
- `tests/` — synthetic and unit tests.

The uploaded dataset loaders were corrected so an ES image cannot silently receive the first available ED ground-truth mask.

## Environment

Use the PyTorch/CUDA environment that already runs your DenseNet code. Install the remaining dependencies:

```bash
python -m pip install -r requirements_scag.txt
```

Install PyTorch and torchvision separately using the builds matching your CUDA version if they are not already installed.

## Step 1 — Pull the branch and run tests

```bash
git fetch origin
git switch hermes/independent-alignment
python -m pytest tests -q
```

Expected result in the tested version: all tests pass.

## Step 2 — Synthetic verification (no medical data)

```bash
bash run_scag_gpu.sh simulate
```

This creates aligned and null synthetic experiments under:

```text
results_mnm2/simulation_verification/
├── aligned/
└── null/
```

The aligned simulation should produce a clearly larger positive `mean_edge_minus_matched_nonedge` effect than the null simulation. Exact values are seed-dependent.

## Step 3 — Small real-data smoke test

```bash
NUM_WORKERS=0 BATCH_SIZE=8 bash run_scag_gpu.sh smoke-data
```

This processes 32 LA samples for cropped three-concept Block 1 and then runs a small three-fold analysis. It verifies loader, checkpoint, hook, IoU, NPZ, CUDA correlation, and result-output integration. It is not a publication experiment.

If CUDA memory is limited:

```bash
CHANNEL_CHUNK=8 BATCH_SIZE=4 SPEARMAN_CHUNK=256 NUM_WORKERS=0 \
  bash run_scag_gpu.sh smoke-data
```

## Step 4 — Primary three-concept experiment

```bash
bash run_scag_gpu.sh primary
```

This runs cropped and full images for Blocks 1–4 using the existing defaults:

```text
M&Ms-2 root: /media/kislay/New Volume/Turab/data/MnM2/
Checkpoint:   /media/kislay/New Volume/Turab/CRAFT/models/best_densenet161_MnMs.pth
```

Override without editing code:

```bash
MNM2_ROOT="/your/MnM2/path/" \
MODEL_PATH="/your/checkpoint.pth" \
DEVICE="cuda:0" \
  bash run_scag_gpu.sh primary
```

The primary run defaults to:

- 5 patient folds;
- densities 1%, 2.5%, 5%, and 10%;
- 1,000 profile permutations;
- 100 degree-preserving rewired graphs;
- 1,000 patient bootstraps.

For an exploratory run before the final computation:

```bash
PERMUTATIONS=100 REWIRES=20 BOOTSTRAPS=100 DENSITIES="0.025 0.05" \
  bash run_scag_gpu.sh primary
```

## Step 5 — Nine-region extension

Run only after the primary three-region pipeline succeeds:

```bash
bash run_scag_gpu.sh extended
```

For every results directory, the analysis writes:

- `fold_metrics.csv` — one row per patient fold and graph density;
- `dose_response.csv` — coupling quantile versus held-out semantic similarity;
- `summary.json` — combined fold summary and warnings;
- `analysis_config.json` — exact analysis settings.

The topology-rewiring statistic is a secondary edge-similarity null. The profile-permutation test is applied directly to the declared edge-minus-matched-nonedge primary endpoint and is aggregated across cross-fit folds rather than averaging fold p-values.

## Manual commands

Extraction example:

```bash
python extract_scag_patient_data.py \
  --image-mode cropped --concepts 3 --blocks 1 2 3 4 \
  --device cuda:0 --batch-size 16 --channel-chunk 32
```

Analysis example:

```bash
python scag_independent_alignment.py \
  --input results_mnm2/patient_level/patient_level_3c_cropped_B1.npz \
  --output results_mnm2/independent_3c_cropped_B1 \
  --densities 0.01 0.025 0.05 0.10 \
  --folds 5 --permutations 1000 --rewires 100 --bootstraps 1000 \
  --device cuda:0 --chunk-size 512 --seed 42
```

## What to send back to Hermes

After `primary` finishes, compress and send these directories:

```text
results_mnm2/independent_3c_cropped_B1
results_mnm2/independent_3c_cropped_B2
results_mnm2/independent_3c_cropped_B3
results_mnm2/independent_3c_cropped_B4
results_mnm2/independent_3c_full_B1
results_mnm2/independent_3c_full_B2
results_mnm2/independent_3c_full_B3
results_mnm2/independent_3c_full_B4
```

Also send the extraction metadata JSON files from `results_mnm2/patient_level/`. The large NPZ files are not initially necessary unless a result needs debugging.

If the analysis stops because fewer than 70% of graph edges have exact degree/activity-stratum non-edge matches, first reduce graph density. Do **not** automatically lower `--min-match-rate`; any lower value becomes an explicit sensitivity analysis and the reduced matched-edge coverage must be reported.

## Interpretation rules

A potentially strong result requires all of the following:

1. positive edge-minus-matched-nonedge effect;
2. patient-bootstrap interval that does not substantially overlap zero;
3. profile-permutation and topology-null evidence;
4. a monotonic or interpretable coupling–semantic dose response;
5. stability across several graph densities and patient folds;
6. preferably replication across blocks, checkpoints, and another architecture.

Do not treat high purity/NMI/ARI of semantically selected components as independent evidence. Global NMI/ARI in the new output are descriptive tests of unsupervised graph communities only.

## Remaining limitation

This pipeline separates graph construction from semantic evaluation at the patient level, but it cannot determine whether the classifier itself was trained on evaluation patients unless the original classifier split file is supplied. That split must be recovered before making a fully leakage-free publication claim.
