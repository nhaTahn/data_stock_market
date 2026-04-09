#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/venv/bin/python"
RUN_BASE="$ROOT/data/processed/assets/data_info_vn/history/training_runs"
DATA_PATH="$ROOT/data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
LOG_DIR="$RUN_BASE/overnight_logs/$(date +%Y%m%d_%H%M%S)_whole_market_watchlist"
MANIFEST="$LOG_DIR/whole_market_manifest.csv"

mkdir -p "$LOG_DIR"
echo "kind,run_name,context_run,expert_run,output_name,preset,sector,window,label" > "$MANIFEST"

run_logged() {
  local log_name="$1"
  shift
  echo "[run] $log_name"
  "$@" > "$LOG_DIR/$log_name.log" 2>&1
}

sector_stocks() {
  local sector_name="$1"
  SECTOR_NAME="$sector_name" DATA_PATH="$DATA_PATH" python3 - <<'PY'
import os
import pandas as pd

sector = os.environ["SECTOR_NAME"]
data_path = os.environ["DATA_PATH"]
df = pd.read_csv(data_path)
codes = sorted(df.loc[df["sector"] == sector, "code"].dropna().astype(str).unique().tolist())
print(",".join(codes))
PY
}

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

STAMP="$(basename "$LOG_DIR" | cut -d_ -f1,2)"
SHARED_SEEDS="42,52,62,72,82,92"
SECTOR_SEEDS="42,52,62,72,82,92"
SHARED_NAME_W20="overnight_wholemarket_shared_vn100_w20_${STAMP}"
SHARED_NAME_W60="overnight_wholemarket_shared_vn100_w60_${STAMP}"

run_logged "shared_vn100_w20" \
  "$PYTHON_BIN" "$ROOT/scripts/run_shared_vn30_committee.py" \
  --context-run-dir "$RUN_BASE/$SHARED_NAME_W20" \
  --universe-path "$ROOT/market_lists/vn100.txt" \
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
  --skip-committee
record_row "shared_run" "$SHARED_NAME_W20" "" "" "" "" "VN100" "20" "shared_vn100_w20"

run_logged "shared_vn100_w60" \
  "$PYTHON_BIN" "$ROOT/scripts/run_shared_vn30_committee.py" \
  --context-run-dir "$RUN_BASE/$SHARED_NAME_W60" \
  --universe-path "$ROOT/market_lists/vn100.txt" \
  --window-size 60 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds "$SHARED_SEEDS" \
  --sample-weight-mode none \
  --skip-committee
record_row "shared_run" "$SHARED_NAME_W60" "" "" "" "" "VN100" "60" "shared_vn100_w60"

SECTORS=(
  "Ngân hàng|bank"
  "Dịch vụ tài chính|dich_vu_tai_chinh"
  "Xây dựng và Vật liệu|xay_dung_va_vat_lieu"
  "Điện, nước & xăng dầu khí đốt|dien_nuoc_xang_dau_khi_dot"
)

for item in "${SECTORS[@]}"; do
  IFS='|' read -r sector_name sector_slug <<< "$item"
  stocks="$(sector_stocks "$sector_name")"
  run_name="overnight_sector_${sector_slug}_w20_${STAMP}"
  run_logged "sector_${sector_slug}_w20" \
    "$PYTHON_BIN" "$ROOT/scripts/run_train.py" \
    --run-name "$run_name" \
    --target-mode return \
    --loss rel_score \
    --sector "$sector_name" \
    --stocks "$stocks" \
    --feature-selection-mode sector_config \
    --window-size 20 \
    --lstm-units 64,32 \
    --dropout 0.05 \
    --lr 0.0005 \
    --batch-size 64 \
    --epochs 30 \
    --patience 8 \
    --target-normalizer volatility_20 \
    --lstm-seeds "$SECTOR_SEEDS" \
    --sample-weight-mode none
  record_row "sector_run" "$run_name" "" "" "" "" "$sector_name" "20" "sector_${sector_slug}_w20"

  for context_run in "$SHARED_NAME_W20" "$SHARED_NAME_W60"; do
    context_window="20"
  if [[ "$context_run" == *"_w60_"* ]]; then
      context_window="60"
    fi
    output_name="${context_run}__committee__${sector_slug}"
    run_logged "committee_${sector_slug}_${context_window}" \
      "$PYTHON_BIN" "$ROOT/src/research/committee_relscore_experiment.py" \
      --expert-run "$RUN_BASE/$run_name" \
      --expert-models "lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble" \
      --market-run "$RUN_BASE/$context_run" \
      --market-models "lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble" \
      --methods "avg,agree_only" \
      --weight-step 0.05 \
      --stable-weight-tolerance 0.001 \
      --selection-mode stable_band \
      --stable-selection-val-gap 0.006 \
      --stable-selection-min-weight-count 2 \
      --output-name "$output_name"
    record_row "committee" "" "${context_run}" "${run_name}" "${output_name}" "${sector_slug}" "${sector_name}" "${context_window}" "committee_${sector_slug}_w${context_window}"
  done
done

run_logged "summarize_whole_market_watchlist" \
  "$PYTHON_BIN" "$ROOT/src/research/summarize_whole_market_watchlist.py" \
  --manifest "$MANIFEST"

echo "Whole-market watchlist manifest: $MANIFEST"
