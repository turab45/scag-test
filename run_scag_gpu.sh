#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_scag_gpu.sh simulate
#   bash run_scag_gpu.sh smoke-data
#   bash run_scag_gpu.sh primary    # 3 concepts, cropped + full, B1-B4
#   bash run_scag_gpu.sh extended   # 9 concepts, cropped + full, B1-B4
#   bash run_scag_gpu.sh all

MODE="${1:-primary}"
PYTHON="${PYTHON:-python}"
MNM2_ROOT="${MNM2_ROOT:-/media/kislay/New Volume/Turab/data/MnM2/}"
MODEL_PATH="${MODEL_PATH:-/media/kislay/New Volume/Turab/CRAFT/models/best_densenet161_MnMs.pth}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_ROOT="${RESULTS_ROOT:-results_mnm2/patient_level}"
FOLDS="${FOLDS:-5}"
PERMUTATIONS="${PERMUTATIONS:-1000}"
REWIRES="${REWIRES:-100}"
BOOTSTRAPS="${BOOTSTRAPS:-1000}"
DENSITIES="${DENSITIES:-0.01 0.025 0.05 0.10}"
SEED="${SEED:-42}"

run_simulation() {
  "$PYTHON" scag_independent_alignment.py \
    --simulate both \
    --output results_mnm2/simulation_verification \
    --densities 0.05 0.10 \
    --folds 3 --quick --device cpu --seed "$SEED"
}

extract_variant() {
  local concepts="$1"
  local image_mode="$2"
  local max_images="${3:-0}"
  local blocks=(1 2 3 4)
  if [[ "$max_images" != "0" ]]; then
    blocks=(1)
  fi
  "$PYTHON" extract_scag_patient_data.py \
    --image-mode "$image_mode" --concepts "$concepts" \
    --blocks "${blocks[@]}" \
    --mnm2-root "$MNM2_ROOT" --model-path "$MODEL_PATH" \
    --output-dir "$RESULTS_ROOT" --device "$DEVICE" \
    --batch-size "${BATCH_SIZE:-16}" --num-workers "${NUM_WORKERS:-4}" \
    --channel-chunk "${CHANNEL_CHUNK:-32}" --max-images "$max_images"
}

analyse_variant() {
  local concepts="$1"
  local image_mode="$2"
  local blocks=(1 2 3 4)
  if [[ "${3:-0}" != "0" ]]; then
    blocks=(1)
  fi
  for block in "${blocks[@]}"; do
    local input="$RESULTS_ROOT/patient_level_${concepts}c_${image_mode}_B${block}.npz"
    local output="results_mnm2/independent_${concepts}c_${image_mode}_B${block}"
    "$PYTHON" scag_independent_alignment.py \
      --input "$input" --output "$output" \
      --densities $DENSITIES --folds "$FOLDS" \
      --permutations "$PERMUTATIONS" --rewires "$REWIRES" \
      --rewire-swap-factor 1 --bootstraps "$BOOTSTRAPS" \
      --device "$DEVICE" --chunk-size "${SPEARMAN_CHUNK:-512}" --seed "$SEED"
  done
}

case "$MODE" in
  simulate)
    run_simulation
    ;;
  smoke-data)
    run_simulation
    FOLDS=3 PERMUTATIONS=30 REWIRES=10 BOOTSTRAPS=30 DENSITIES="0.05" \
      extract_variant 3 cropped 32
    FOLDS=3 PERMUTATIONS=30 REWIRES=10 BOOTSTRAPS=30 DENSITIES="0.05" \
      analyse_variant 3 cropped 32
    ;;
  primary)
    run_simulation
    for image_mode in cropped full; do
      extract_variant 3 "$image_mode"
      analyse_variant 3 "$image_mode"
    done
    ;;
  extended)
    for image_mode in cropped full; do
      extract_variant 9 "$image_mode"
      analyse_variant 9 "$image_mode"
    done
    ;;
  all)
    run_simulation
    for concepts in 3 9; do
      for image_mode in cropped full; do
        extract_variant "$concepts" "$image_mode"
        analyse_variant "$concepts" "$image_mode"
      done
    done
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

echo "SCAG run completed: mode=$MODE"
