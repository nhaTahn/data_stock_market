# VN Improvement Smoke

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)

## Aggregate (sorted by rel_score_mean)

| variant       |   rel_score_mean |   rel_score_std |   DA_mean |   n_seeds | description                                        |
|:--------------|-----------------:|----------------:|----------:|----------:|:---------------------------------------------------|
| rich_wrel085  |         0.029006 |        0.004034 |  0.508009 |         5 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |
| rich_rel_only |         0.027643 |        0.00611  |  0.507205 |         5 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |

## Per Seed

| variant       |   seed |   rel_score |   directional_accuracy |   pred_actual_q90_ratio | description                                        |
|:--------------|-------:|------------:|-----------------------:|------------------------:|:---------------------------------------------------|
| rich_rel_only |     43 |    0.018738 |               0.501464 |                0.151922 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |
| rich_rel_only |     52 |    0.034818 |               0.511573 |                0.200891 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |
| rich_rel_only |     62 |    0.029109 |               0.50962  |                0.220785 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |
| rich_rel_only |     71 |    0.030634 |               0.507933 |                0.192558 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |
| rich_rel_only |     82 |    0.024918 |               0.505435 |                0.166529 | DEFAULT + rich feats, [64,32], w=15, rel_only_loss |
| rich_wrel085  |     43 |    0.025552 |               0.505484 |                0.185135 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |
| rich_wrel085  |     52 |    0.034818 |               0.511573 |                0.200891 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |
| rich_wrel085  |     62 |    0.029109 |               0.50962  |                0.220785 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |
| rich_wrel085  |     71 |    0.030634 |               0.507933 |                0.192558 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |
| rich_wrel085  |     82 |    0.024918 |               0.505435 |                0.166529 | DEFAULT + rich feats, [64,32], w=15, w_rel=0.85    |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/vn_loss_ablation_5seed_20260526",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/vn_loss_ablation_5seed_20260526"
}