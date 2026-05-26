# VN Improvement Smoke

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)

## Aggregate (sorted by rel_score_mean)

| variant   |   rel_score_mean |   rel_score_std |   DA_mean |   n_seeds | description                                                             |
|:----------|-----------------:|----------------:|----------:|----------:|:------------------------------------------------------------------------|
| rich_feat |         0.029943 |        0.003546 |  0.508644 |         5 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |

## Per Seed

| variant   |   seed |   rel_score |   directional_accuracy |   pred_actual_q90_ratio | description                                                             |
|:----------|-------:|------------:|-----------------------:|------------------------:|:------------------------------------------------------------------------|
| rich_feat |     43 |    0.030238 |               0.508661 |                0.209983 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     52 |    0.034818 |               0.511573 |                0.200891 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     62 |    0.029109 |               0.50962  |                0.220785 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     71 |    0.030634 |               0.507933 |                0.192558 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     82 |    0.024918 |               0.505435 |                0.166529 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/vn_rich_feat_full5_preds_20260526",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/vn_rich_feat_full5_preds_20260526"
}