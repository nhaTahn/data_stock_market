# VN Improvement Smoke

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)

## Aggregate (sorted by rel_score_mean)

| variant     |   rel_score_mean |   rel_score_std |   DA_mean |   n_seeds | description                               |
|:------------|-----------------:|----------------:|----------:|----------:|:------------------------------------------|
| rich_deeper |         0.024293 |        0.000983 |  0.507199 |         3 | DEFAULT + rich feats, [96,64,32]+LN, w=15 |

## Per Seed

| variant     |   seed |   rel_score |   directional_accuracy |   pred_actual_q90_ratio | description                               |
|:------------|-------:|------------:|-----------------------:|------------------------:|:------------------------------------------|
| rich_deeper |     43 |    0.025017 |               0.510117 |                0.152377 | DEFAULT + rich feats, [96,64,32]+LN, w=15 |
| rich_deeper |     52 |    0.023174 |               0.512069 |                0.185662 | DEFAULT + rich feats, [96,64,32]+LN, w=15 |
| rich_deeper |     62 |    0.024687 |               0.499413 |                0.15562  | DEFAULT + rich feats, [96,64,32]+LN, w=15 |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/vn_rich_deeper_3seed_20260526",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/vn_rich_deeper_3seed_20260526"
}