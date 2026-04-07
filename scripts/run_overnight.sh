#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export TF_CPP_MIN_LOG_LEVEL=2

STAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_BASE="data/processed/assets/data_info_vn/history/training_runs"
LOG_DIR="${RUN_BASE}/overnight_logs/${STAMP}"
mkdir -p "${LOG_DIR}"

declare -a RUN_NAMES=()

run_step() {
  local step_name="$1"
  shift
  echo ""
  echo ">>> ${step_name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step_name}.log"
}

train_and_post() {
  local run_name="$1"
  shift
  run_step "${run_name}_train" venv/bin/python scripts/run_train.py "$@" --run-name "${run_name}"
  run_step "${run_name}_backtest" venv/bin/python src/models/backtest_threshold.py "${RUN_BASE}/${run_name}" --non-overlap
  run_step "${run_name}_report" venv/bin/python src/models/update_run_reports.py "${RUN_BASE}/${run_name}"
  RUN_NAMES+=("${run_name}")
}

echo "==========================================="
echo "BAT DAU CHUOI NHIEM VU CHAY QUA DEM"
echo "Log dir: ${LOG_DIR}"
echo "==========================================="

run_step "build_vn_dataset" venv/bin/python scripts/run_build_dataset.py --market VN

run_step "search_vhm_return3d_default" \
  venv/bin/python src/models/search_feature_combinations.py \
  --target-mode return_3d \
  --stocks VHM \
  --window-size 5 \
  --min-combo-size 2 \
  --max-combo-size 3 \
  --min-rel-score 0.01 \
  --top-k 40 \
  --run-name search_vhm_return3d_default_overnight

run_step "search_nlg_return3d_default" \
  venv/bin/python src/models/search_feature_combinations.py \
  --target-mode return_3d \
  --stocks NLG \
  --window-size 5 \
  --min-combo-size 2 \
  --max-combo-size 3 \
  --min-rel-score 0.01 \
  --top-k 40 \
  --run-name search_nlg_return3d_default_overnight

run_step "search_dig_return3d_default" \
  venv/bin/python src/models/search_feature_combinations.py \
  --target-mode return_3d \
  --stocks DIG \
  --window-size 5 \
  --min-combo-size 2 \
  --max-combo-size 3 \
  --min-rel-score 0.01 \
  --top-k 40 \
  --run-name search_dig_return3d_default_overnight

train_and_post \
  "overnight_bds_return3d_sector_u16_w7" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --window-size 7 \
  --lstm-units 16 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 35 \
  --patience 8

train_and_post \
  "overnight_bds_return3d_sector_u32_w10" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --window-size 10 \
  --lstm-units 32 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10

train_and_post \
  "overnight_bds_return3d_sector_u48_w12" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --window-size 12 \
  --lstm-units 48 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10

train_and_post \
  "overnight_bds_return3d_sector_stack48_24_w10" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --window-size 10 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 45 \
  --patience 12

train_and_post \
  "overnight_bds_return3d_allfeat_u16_w7" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --use-all-features \
  --window-size 7 \
  --lstm-units 16 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 35 \
  --patience 8

train_and_post \
  "overnight_bds_return3d_allfeat_u32_w10" \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --use-all-features \
  --window-size 10 \
  --lstm-units 32 \
  --dropout 0.05 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 40 \
  --patience 10

train_and_post \
  "overnight_vhm_return3d_pairmacro_w5_u16" \
  --target-mode return_3d \
  --stocks VHM \
  --feature-columns upper_shadow,bb_width,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 5 \
  --lstm-units 16 \
  --dropout 0.05 \
  --lr 0.0005 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

train_and_post \
  "overnight_vhm_return3d_pairmacro_w3_u8" \
  --target-mode return_3d \
  --stocks VHM \
  --feature-columns upper_shadow,bb_width,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 3 \
  --lstm-units 8 \
  --dropout 0.0 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

train_and_post \
  "overnight_nlg_return3d_pairmacro_w5_u8" \
  --target-mode return_3d \
  --stocks NLG \
  --feature-columns upper_shadow,momentum_20,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 5 \
  --lstm-units 8 \
  --dropout 0.0 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

train_and_post \
  "overnight_nlg_return3d_pairmacro_w5_u16" \
  --target-mode return_3d \
  --stocks NLG \
  --feature-columns upper_shadow,momentum_20,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 5 \
  --lstm-units 16 \
  --dropout 0.05 \
  --lr 0.0005 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

train_and_post \
  "overnight_dig_return3d_pairmacro_w3_u8" \
  --target-mode return_3d \
  --stocks DIG \
  --feature-columns gap_open,bb_position,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 3 \
  --lstm-units 8 \
  --dropout 0.0 \
  --lr 0.0003 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

train_and_post \
  "overnight_dig_return3d_pairmacro_w5_u16" \
  --target-mode return_3d \
  --stocks DIG \
  --feature-columns gap_open,bb_position,vingroup_momentum,vnindex_return,rsi_14,day_of_week \
  --window-size 5 \
  --lstm-units 16 \
  --dropout 0.05 \
  --lr 0.0005 \
  --loss huber \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 30 \
  --patience 8

run_step "overnight_summary" venv/bin/python - "${RUN_BASE}" "${LOG_DIR}/overnight_summary.csv" "${RUN_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

run_base = Path(sys.argv[1])
out_path = Path(sys.argv[2])
run_names = sys.argv[3:]

rows = []
for run_name in run_names:
    metrics_path = run_base / run_name / "metrics.json"
    backtest_path = run_base / run_name / "threshold_backtest_summary_non_overlap.json"
    if not metrics_path.exists():
        continue

    metrics = json.loads(metrics_path.read_text())
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
    lstm_test = metrics.get("lstm", {}).get("test", {})
    linear_test = metrics.get("linear_regression", {}).get("test", {})
    arima_test = metrics.get("arima", {}).get("test", {})
    lstm_bt = backtest.get("lstm", {})

    rows.append(
        {
            "run_name": run_name,
            "lstm_test_rel_score": lstm_test.get("rel_score"),
            "lstm_test_directional_accuracy": lstm_test.get("directional_accuracy"),
            "linear_test_rel_score": linear_test.get("rel_score"),
            "arima_test_rel_score": arima_test.get("rel_score"),
            "lstm_backtest_final_equity": lstm_bt.get("final_equity"),
            "lstm_backtest_trade_count": lstm_bt.get("trade_count"),
            "lstm_backtest_threshold": lstm_bt.get("threshold"),
        }
    )

summary_df = pd.DataFrame(rows).sort_values(
    ["lstm_test_rel_score", "lstm_backtest_final_equity"],
    ascending=[False, False],
)
summary_df.to_csv(out_path, index=False)
print(summary_df.to_string(index=False))
print(f"Saved: {out_path}")
PY

run_step "overnight_archive_candidates" \
  venv/bin/python src/models/archive_lstm_candidates.py \
  --run-base "${RUN_BASE}" \
  --run-names "${RUN_NAMES[@]}" \
  --threshold 0.03 \
  --summary-csv "${LOG_DIR}/overnight_lstm_summary.csv" \
  --candidates-csv "${LOG_DIR}/overnight_lstm_candidates_ge_003.csv"

echo ""
echo "==========================================="
echo "HOAN TAT TOAN BO"
echo "Bao cao va log da xuat tai: ${RUN_BASE}"
echo "Log chi tiet: ${LOG_DIR}"
echo "==========================================="
