#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/venv/bin/python"
RUN_BASE="$ROOT/data/processed/assets/data_info_vn/history/training_runs"
LOG_DIR="$RUN_BASE/overnight_logs/$(date +%Y%m%d_%H%M%S)_relscore_push"
MANIFEST="$LOG_DIR/overnight_manifest.csv"

mkdir -p "$LOG_DIR"
echo "kind,run_name,preset,label" > "$MANIFEST"

run_logged() {
  local log_name="$1"
  shift
  echo "[run] $log_name"
  "$@" > "$LOG_DIR/$log_name.log" 2>&1
}

record_train() {
  local run_name="$1"
  local label="$2"
  echo "train,$run_name,,$label" >> "$MANIFEST"
}

record_committee() {
  local run_name="$1"
  local preset="$2"
  local label="$3"
  echo "shared_committee,$run_name,$preset,$label" >> "$MANIFEST"
}

STAMP="$(basename "$LOG_DIR" | cut -d_ -f1,2)"

FNB_FEATURES="volume_ratio_20,close_position,lower_shadow,alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"
BDS_FEATURES="close_position,momentum_20,gap_open,vwap_gap,bb_width,above_ma_200,upper_shadow,alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"

FNB_SEEDS="42,52,62,72,82,92"
BDS_SEEDS="42,52,62,72,82,92,99,109"
SHARED_SEEDS="42,52,62,72,82,92"

FNB_STOCKS="KDC,SAB,SBT,VNM"
BDS_STOCKS="KOS,DXG,NLG,DIG,TCH,VHM"

run_logged "fnb_w5_plain_more_seeds" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_fnb_w5_plain_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$FNB_STOCKS" \
  --feature-columns "$FNB_FEATURES" \
  --window-size 5 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$FNB_SEEDS" \
  --sample-weight-mode none
record_train "overnight_relpush_fnb_w5_plain_${STAMP}" "fnb_w5_plain_more_seeds"

run_logged "fnb_w5_plain_magweight" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_fnb_w5_magweight_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$FNB_STOCKS" \
  --feature-columns "$FNB_FEATURES" \
  --window-size 5 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$FNB_SEEDS" \
  --sample-weight-mode magnitude
record_train "overnight_relpush_fnb_w5_magweight_${STAMP}" "fnb_w5_plain_magweight"

run_logged "fnb_w10_plain_more_seeds" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_fnb_w10_plain_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$FNB_STOCKS" \
  --feature-columns "$FNB_FEATURES" \
  --window-size 10 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$FNB_SEEDS" \
  --sample-weight-mode none
record_train "overnight_relpush_fnb_w10_plain_${STAMP}" "fnb_w10_plain_more_seeds"

run_logged "fnb_w10_plain_magweight" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_fnb_w10_magweight_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$FNB_STOCKS" \
  --feature-columns "$FNB_FEATURES" \
  --window-size 10 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$FNB_SEEDS" \
  --sample-weight-mode magnitude
record_train "overnight_relpush_fnb_w10_magweight_${STAMP}" "fnb_w10_plain_magweight"

run_logged "bds_w20_more_seeds" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_bds_w20_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$BDS_STOCKS" \
  --feature-columns "$BDS_FEATURES" \
  --window-size 20 \
  --lstm-units 64,32 \
  --dropout 0.08 \
  --lr 0.0003 \
  --batch-size 64 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$BDS_SEEDS" \
  --sample-weight-mode none
record_train "overnight_relpush_bds_w20_${STAMP}" "bds_w20_more_seeds"

run_logged "bds_w15_more_seeds" \
  "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
  --run-name "overnight_relpush_bds_w15_${STAMP}" \
  --target-mode return \
  --loss rel_score \
  --stocks "$BDS_STOCKS" \
  --feature-columns "$BDS_FEATURES" \
  --window-size 15 \
  --lstm-units 64,32 \
  --dropout 0.10 \
  --lr 0.0003 \
  --batch-size 64 \
  --epochs 40 \
  --patience 10 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$BDS_SEEDS" \
  --sample-weight-mode none
record_train "overnight_relpush_bds_w15_${STAMP}" "bds_w15_more_seeds"

run_logged "shared_vn100_w20_fnb_committee_more_seeds" \
  "$PYTHON_BIN" "$ROOT/scripts/run_shared_vn30_committee.py" \
  --run-name-prefix "overnight_relpush_shared_vn100_w20" \
  --universe-path "$ROOT/market_lists/vn100.txt" \
  --committee-preset fnb \
  --window-size 20 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$SHARED_SEEDS" \
  --sample-weight-mode none \
  --committee-selection-mode stable_band \
  --committee-stable-selection-val-gap 0.006 \
  --committee-stable-selection-min-weight-count 2 \
  --rotation-min-val-rel-score 0.015 \
  --rotation-min-stable-weight-count 2

LATEST_SHARED_RUN="$(python3 - <<PY
from pathlib import Path
base=Path('$RUN_BASE')
cands=sorted(base.glob('overnight_relpush_shared_vn100_w20_*'))
print(cands[-1].name if cands else '')
PY
)"
if [[ -n "$LATEST_SHARED_RUN" ]]; then
  record_committee "$LATEST_SHARED_RUN" "fnb" "shared_vn100_w20_fnb_committee_more_seeds"
fi

run_logged "summarize_relscore_push_overnight" \
  "$PYTHON_BIN" "$ROOT/src/research/summarize_relscore_push_overnight.py" \
  --manifest "$MANIFEST"

echo "Overnight plan manifest: $MANIFEST"
