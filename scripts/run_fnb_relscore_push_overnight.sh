#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/venv/bin/python"
RUN_BASE="$ROOT/data/processed/assets/data_info_vn/history/training_runs"
LOG_DIR="$RUN_BASE/overnight_logs/$(date +%Y%m%d_%H%M%S)_fnb_relscore_push"
MANIFEST="$LOG_DIR/fnb_manifest.csv"
SHARED_CONTEXT_RUN="confirm_vn100_fnb_committee_20260408_235445_r01"

mkdir -p "$LOG_DIR"
echo "kind,run_name,output_name,committee_kind,label" > "$MANIFEST"

record_row() {
  python3 - "$MANIFEST" "$@" <<'PY'
import csv
import sys

manifest = sys.argv[1]
fields = sys.argv[2:]
with open(manifest, "a", encoding="utf-8", newline="") as handle:
    csv.writer(handle).writerow(fields)
PY
}

run_logged() {
  local log_name="$1"
  shift
  echo "[run] $log_name"
  "$@" > "$LOG_DIR/$log_name.log" 2>&1
}

run_internal_committee() {
  local run_name="$1"
  local label="$2"
  local output_name="${run_name}__committee__internal_bias_fix"
  run_logged "committee_internal_${run_name}" \
    "$PYTHON_BIN" "$ROOT/src/research/committee_relscore_experiment.py" \
    --expert-run "$RUN_BASE/$run_name" \
    --expert-models "lstm_best_by_val,lstm_ensemble" \
    --market-run "$RUN_BASE/$run_name" \
    --market-models "lstm_signmag_best_by_val,lstm_signmag_top2_by_val,lstm_signmag_ensemble,lstm_quantile_best_by_val" \
    --methods "avg,agree_only" \
    --weight-step 0.05 \
    --stable-weight-tolerance 0.001 \
    --selection-mode stable_band \
    --stable-selection-val-gap 0.006 \
    --stable-selection-min-weight-count 2 \
    --output-name "$output_name"
  record_row "committee" "" "$output_name" "internal" "${label}_internal"
}

run_shared_committee() {
  local run_name="$1"
  local label="$2"
  local output_name="${run_name}__committee__shared_vn100"
  run_logged "committee_shared_${run_name}" \
    "$PYTHON_BIN" "$ROOT/src/research/committee_relscore_experiment.py" \
    --expert-run "$RUN_BASE/$run_name" \
    --expert-models "lstm_best_by_val,lstm_ensemble" \
    --market-run "$RUN_BASE/$SHARED_CONTEXT_RUN" \
    --market-models "lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble" \
    --methods "avg,agree_only" \
    --weight-step 0.05 \
    --stable-weight-tolerance 0.001 \
    --selection-mode stable_band \
    --stable-selection-val-gap 0.006 \
    --stable-selection-min-weight-count 2 \
    --output-name "$output_name"
  record_row "committee" "" "$output_name" "shared_vn100" "${label}_shared"
}

run_fnb_case() {
  local run_name="$1"
  local label="$2"
  local window_size="$3"
  local lstm_units="$4"
  local lr="$5"
  local dropout="$6"
  local sample_weight_mode="$7"
  local sample_weight_strength="$8"
  local sample_weight_quantile="$9"
  local sample_weight_clip="${10}"

  cmd=(
    "$PYTHON_BIN" "$ROOT/scripts/run_train.py"
    --run-name "$run_name"
    --target-mode return
    --loss rel_score
    --stocks "KDC,SAB,SBT,VNM"
    --feature-columns "volume_ratio_20,close_position,lower_shadow,alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"
    --window-size "$window_size"
    --lstm-units "$lstm_units"
    --dropout "$dropout"
    --lr "$lr"
    --batch-size 32
    --epochs 40
    --patience 10
    --target-normalizer volatility_20
    --lstm-seeds "42,52,62,72,82,92,102,112"
    --sample-weight-mode "$sample_weight_mode"
  )
  if [[ "$sample_weight_mode" == "magnitude" ]]; then
    cmd+=(--sample-weight-strength "$sample_weight_strength" --sample-weight-quantile "$sample_weight_quantile" --sample-weight-clip "$sample_weight_clip")
  fi

  run_logged "$label" "${cmd[@]}"
  record_row "train" "$run_name" "" "" "$label"
  run_internal_committee "$run_name" "$label"
  run_shared_committee "$run_name" "$label"
}

STAMP="$(basename "$LOG_DIR" | cut -d_ -f1,2)"

run_fnb_case "overnight_fnb_w5_mag_base_${STAMP}" "fnb_w5_mag_base" "5" "48,24" "0.0002" "0.05" "magnitude" "0.8" "0.8" "2.5"
run_fnb_case "overnight_fnb_w5_mag_tighter_lr_${STAMP}" "fnb_w5_mag_tighter_lr" "5" "48,24" "0.00015" "0.05" "magnitude" "0.8" "0.8" "2.5"
run_fnb_case "overnight_fnb_w5_mag_higher_units_${STAMP}" "fnb_w5_mag_higher_units" "5" "64,32" "0.0002" "0.05" "magnitude" "0.8" "0.8" "2.5"
run_fnb_case "overnight_fnb_w5_nomag_${STAMP}" "fnb_w5_nomag" "5" "48,24" "0.0002" "0.05" "none" "" "" ""
run_fnb_case "overnight_fnb_w7_mag_${STAMP}" "fnb_w7_mag" "7" "48,24" "0.0002" "0.05" "magnitude" "0.8" "0.8" "2.5"
run_fnb_case "overnight_fnb_w10_mag_${STAMP}" "fnb_w10_mag" "10" "48,24" "0.0002" "0.05" "magnitude" "0.8" "0.8" "2.5"

run_logged "summarize_fnb_relscore_push" \
  "$PYTHON_BIN" "$ROOT/src/research/summarize_fnb_relscore_push.py" \
  --manifest "$MANIFEST"

echo "F&B relscore push manifest: $MANIFEST"
