# VN Improvement Smoke

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)

## Aggregate (sorted by rel_score_mean)

| variant   |   rel_score_mean |   rel_score_std |   DA_mean |   n_seeds | description                                                             |
|:----------|-----------------:|----------------:|----------:|----------:|:------------------------------------------------------------------------|
| rich_feat |         0.022258 |         0.00662 |  0.504469 |         5 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |

## Per Seed

| variant   |   seed |   rel_score |   directional_accuracy |   pred_actual_q90_ratio | description                                                             |
|:----------|-------:|------------:|-----------------------:|------------------------:|:------------------------------------------------------------------------|
| rich_feat |     43 |    0.01627  |               0.503582 |                0.138496 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     52 |    0.030833 |               0.509025 |                0.17464  | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     62 |    0.022628 |               0.501977 |                0.165739 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     71 |    0.026318 |               0.507652 |                0.161204 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     82 |    0.015239 |               0.500108 |                0.130544 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/vn_rich_feat_d002_5seed_20260526",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/vn_rich_feat_d002_5seed_20260526"
}