#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export TF_CPP_MIN_LOG_LEVEL=2

STAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_BASE="data/processed/assets/data_info_vn/history/training_runs"
LOG_DIR="${RUN_BASE}/overnight_logs/${STAMP}_shared_context"
mkdir -p "${LOG_DIR}"

declare -a CONTEXT_RUNS=()

run_step() {
  local step_name="$1"
  shift
  echo ""
  echo ">>> ${step_name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step_name}.log"
}

run_suite() {
  local run_name="$1"
  shift
  run_step "${run_name}" venv/bin/python scripts/run_shared_vn30_committee.py \
    --context-run-dir "${RUN_BASE}/${run_name}" \
    --committee-preset bank \
    --committee-preset bds \
    --committee-preset fnb \
    "$@"
  CONTEXT_RUNS+=("${run_name}")
}

echo "==========================================="
echo "BAT DAU SHARED CONTEXT OVERNIGHT"
echo "Log dir: ${LOG_DIR}"
echo "==========================================="

run_step "build_vn_dataset" venv/bin/python scripts/run_build_dataset.py --market VN

run_suite \
  "overnight_shared_vn30_w20_u64_32_relscore_${STAMP}" \
  --universe-path market_lists/vn30.txt \
  --window-size 20 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62 \
  --sample-weight-mode none

run_suite \
  "overnight_shared_vn30_w60_u64_32_relscore_${STAMP}" \
  --universe-path market_lists/vn30.txt \
  --window-size 60 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62 \
  --sample-weight-mode none

run_suite \
  "overnight_shared_vn100_w20_u64_32_relscore_${STAMP}" \
  --universe-path market_lists/vn100.txt \
  --window-size 20 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62 \
  --sample-weight-mode none

run_suite \
  "overnight_shared_vn100_w60_u64_32_relscore_${STAMP}" \
  --universe-path market_lists/vn100.txt \
  --window-size 60 \
  --lstm-units 64,32 \
  --dropout 0.05 \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 30 \
  --patience 8 \
  --target-normalizer volatility_20 \
  --lstm-seeds 42,52,62 \
  --sample-weight-mode none

run_step "shared_context_overnight_summary" \
  venv/bin/python src/research/summarize_shared_committee_overnight.py \
  --run-base "${RUN_BASE}" \
  --run-names "${CONTEXT_RUNS[@]}" \
  --min-code-count 3 \
  --all-csv "${LOG_DIR}/shared_context_committee_all.csv" \
  --stable-csv "${LOG_DIR}/shared_context_committee_stable_ge3codes.csv"

echo ""
echo "==========================================="
echo "HOAN TAT SHARED CONTEXT OVERNIGHT"
echo "Bao cao va log da xuat tai: ${RUN_BASE}"
echo "Log chi tiet: ${LOG_DIR}"
echo "==========================================="
