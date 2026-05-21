#!/usr/bin/env bash
set -euo pipefail

python3 main.py train \
  --run-name research_hidden_signal_sidecar_20260423_full \
  --feature-columns "volume_ratio_20,intraday_return,gap_open,close_position,upper_shadow,lower_shadow,momentum_5,momentum_20,volatility_20,ma_200_gap,rolling_max_20_gap,bb_width,vwap_gap,obv_change,macd_hist,effort_result_ratio,buying_pressure,selling_pressure,wyckoff_phase_60d,a_d_ratio,vingroup_momentum,vnindex_return,rsi_14,day_of_week,sector_momentum_rank,is_top_2_sector,alpha_sector,vwap_gap_20,above_ma_200"
