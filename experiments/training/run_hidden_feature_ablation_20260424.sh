#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/hidden_feature_ablation_20260424"
mkdir -p "$LOG_DIR"

run_case() {
  local run_name="$1"
  local feature_columns="$2"
  local log_file="$LOG_DIR/${run_name}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $run_name"
  python3 main.py train \
    --run-name "$run_name" \
    --feature-columns "$feature_columns" \
    >"$log_file" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $run_name"
}

BASE_FEATURES="volume_ratio_20,intraday_return,gap_open,close_position,upper_shadow,lower_shadow,momentum_5,momentum_20,volatility_20,ma_200_gap,rolling_max_20_gap,bb_width,vwap_gap,obv_change,macd_hist,effort_result_ratio,buying_pressure,selling_pressure,wyckoff_phase_60d,a_d_ratio,vingroup_momentum,vnindex_return,rsi_14,day_of_week,sector_momentum_rank,is_top_2_sector"

run_case \
  "research_ablate_alpha_20260424" \
  "${BASE_FEATURES},alpha_sector"

run_case \
  "research_ablate_vwap20_20260424" \
  "${BASE_FEATURES},vwap_gap_20"

run_case \
  "research_ablate_alpha_vwap20_20260424" \
  "${BASE_FEATURES},alpha_sector,vwap_gap_20"

run_case \
  "research_ablate_alpha_vwap20_ma200_20260424" \
  "${BASE_FEATURES},alpha_sector,vwap_gap_20,above_ma_200"

python3 experiments/analysis/summarize_hidden_feature_ablation.py \
  --runs \
  research_ablate_alpha_20260424 \
  research_ablate_vwap20_20260424 \
  research_ablate_alpha_vwap20_20260424 \
  research_ablate_alpha_vwap20_ma200_20260424 \
  > "$LOG_DIR/summary.txt"
