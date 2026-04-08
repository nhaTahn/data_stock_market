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
  run_step "${run_name}_backtest" venv/bin/python src/backtesting/threshold_backtest.py "${RUN_BASE}/${run_name}" --non-overlap
  run_step "${run_name}_report" venv/bin/python src/reporting/update_run_reports.py "${RUN_BASE}/${run_name}"
  RUN_NAMES+=("${run_name}")
}

echo "==========================================="
echo "BAT DAU CHUOI NHIEM VU CHAY QUA DEM"
echo "Log dir: ${LOG_DIR}"
echo "==========================================="

run_step "build_vn_dataset" venv/bin/python scripts/run_build_dataset.py --market VN

COMMON_SECTOR="Bất động sản"
COMMON_CONTEXT_FEATURES="alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"

train_and_post \
  "overnight_bds_return1d_stack48_24_w5_core_fk" \
  --target-mode return \
  --sector "${COMMON_SECTOR}" \
  --feature-selection-mode search_summary \
  --feature-top-k 10 \
  --min-stock-val-rel-score 0.03 \
  --max-stocks 10 \
  --extra-context-features "${COMMON_CONTEXT_FEATURES}" \
  --window-size 5 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --loss rel_score \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 24 \
  --patience 6 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62,72,82 \
  --enable-fk-benchmark

train_and_post \
  "overnight_bds_return1d_stack48_24_w5_weighted" \
  --target-mode return \
  --sector "${COMMON_SECTOR}" \
  --feature-selection-mode search_summary \
  --feature-top-k 10 \
  --min-stock-val-rel-score 0.03 \
  --max-stocks 10 \
  --extra-context-features "${COMMON_CONTEXT_FEATURES}" \
  --window-size 5 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --loss rel_score \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 24 \
  --patience 6 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62,72,82 \
  --sample-weight-mode magnitude \
  --sample-weight-strength 0.8 \
  --sample-weight-quantile 0.8 \
  --sample-weight-clip 2.5 \
  --signmag-signed-loss-weight 2.0 \
  --signmag-sign-loss-weight 0.10 \
  --signmag-magnitude-loss-weight 0.25

train_and_post \
  "overnight_bds_return1d_stack48_24_w5_attention" \
  --target-mode return \
  --sector "${COMMON_SECTOR}" \
  --feature-selection-mode search_summary \
  --feature-top-k 10 \
  --min-stock-val-rel-score 0.03 \
  --max-stocks 10 \
  --extra-context-features "${COMMON_CONTEXT_FEATURES}" \
  --window-size 5 \
  --lstm-units 48,24 \
  --dropout 0.05 \
  --lr 0.0002 \
  --loss rel_score \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 24 \
  --patience 6 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,82 \
  --enable-attention-family

train_and_post \
  "overnight_bds_return1d_u32_w5_core" \
  --target-mode return \
  --sector "${COMMON_SECTOR}" \
  --feature-selection-mode search_summary \
  --feature-top-k 10 \
  --min-stock-val-rel-score 0.03 \
  --max-stocks 10 \
  --extra-context-features "${COMMON_CONTEXT_FEATURES}" \
  --window-size 5 \
  --lstm-units 32 \
  --dropout 0.05 \
  --lr 0.0002 \
  --loss rel_score \
  --huber-delta 0.01 \
  --batch-size 32 \
  --epochs 24 \
  --patience 6 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62,72,82

run_step "overnight_summary" venv/bin/python - "${RUN_BASE}" "${LOG_DIR}/overnight_summary.csv" "${RUN_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

run_base = Path(sys.argv[1])
out_path = Path(sys.argv[2])
run_names = sys.argv[3:]


def resolve_artifact(run_dir: Path, filename: str, bucket: str) -> Path:
    candidate = run_dir / "reports" / bucket / filename
    if candidate.exists():
        return candidate
    return run_dir / filename


def resolve_backtest_summary(run_dir: Path, target_mode: str) -> Path:
    if target_mode in {"return_3d", "return_5d"}:
        candidate = resolve_artifact(run_dir, "threshold_backtest_summary_non_overlap.json", "backtests")
        if candidate.exists():
            return candidate
    return resolve_artifact(run_dir, "threshold_backtest_summary.json", "backtests")

rows = []
for run_name in run_names:
    run_dir = run_base / run_name
    metrics_path = resolve_artifact(run_dir, "metrics.json", "core")
    backtest_path = resolve_artifact(run_dir, "threshold_backtest_summary_non_overlap.json", "backtests")
    config_path = resolve_artifact(run_dir, "config.json", "core")
    if not metrics_path.exists():
        continue

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    backtest_path = resolve_backtest_summary(run_dir, config.get("target_mode"))
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
    lstm_test = metrics.get("lstm", {}).get("test", {})
    linear_test = metrics.get("linear_regression", {}).get("test", {})
    arima_test = metrics.get("arima", {}).get("test", {})
    lstm_bt = backtest.get("lstm", {})
    lstm_models = [name for name in metrics if name.startswith("lstm")]
    ranked = sorted(
        lstm_models,
        key=lambda name: (
            metrics.get(name, {}).get("test", {}).get("rel_score", float("-inf")),
            metrics.get(name, {}).get("val", {}).get("rel_score", float("-inf")),
        ),
        reverse=True,
    )
    best_lstm_model = ranked[0] if ranked else None
    best_lstm_test = metrics.get(best_lstm_model, {}).get("test", {}) if best_lstm_model else {}
    best_lstm_val = metrics.get(best_lstm_model, {}).get("val", {}) if best_lstm_model else {}

    rows.append(
        {
            "run_name": run_name,
            "target_mode": config.get("target_mode"),
            "best_lstm_model": best_lstm_model,
            "best_lstm_val_rel_score": best_lstm_val.get("rel_score"),
            "best_lstm_test_rel_score": best_lstm_test.get("rel_score"),
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

run_step "overnight_target_mode_compare" \
  venv/bin/python src/research/compare_target_modes.py \
  --run-base "${RUN_BASE}" \
  --run-names "${RUN_NAMES[@]}" \
  --details-csv "${LOG_DIR}/target_mode_comparison_details.csv" \
  --summary-csv "${LOG_DIR}/target_mode_comparison_summary.csv"

run_step "overnight_archive_candidates" \
  venv/bin/python src/research/archive_lstm_candidates.py \
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
