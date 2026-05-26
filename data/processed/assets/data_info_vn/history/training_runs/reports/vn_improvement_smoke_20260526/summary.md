# VN Improvement Smoke

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)

## Aggregate (sorted by rel_score_mean)

| variant   |   rel_score_mean |   rel_score_std |   DA_mean |   n_seeds | description                                                             |
|:----------|-----------------:|----------------:|----------:|----------:|:------------------------------------------------------------------------|
| rich_feat |         0.021593 |        0.001867 |  0.504773 |         3 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| baseline  |         0.020732 |        0.012822 |  0.501045 |         3 | DEFAULT 26 feats, [64,32], w=15                                         |

## Per Seed

| variant   |   seed |   rel_score |   directional_accuracy |   pred_actual_q90_ratio | description                                                             |
|:----------|-------:|------------:|-----------------------:|------------------------:|:------------------------------------------------------------------------|
| baseline  |     43 |    0.007384 |               0.495988 |                0.112888 | DEFAULT 26 feats, [64,32], w=15                                         |
| baseline  |     52 |    0.032952 |               0.504707 |                0.207937 | DEFAULT 26 feats, [64,32], w=15                                         |
| baseline  |     62 |    0.021861 |               0.50244  |                0.171242 | DEFAULT 26 feats, [64,32], w=15                                         |
| rich_feat |     43 |    0.021794 |               0.500637 |                0.157713 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     52 |    0.019633 |               0.507652 |                0.177904 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |
| rich_feat |     62 |    0.023351 |               0.50603  |                0.171436 | DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15 |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/vn_improvement_smoke_20260526",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/vn_improvement_smoke_20260526"
}