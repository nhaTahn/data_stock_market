# Regime-Conditional Calibration + Conformal Intervals

Protocol: lagged features; validation readout only. Holdout/test not used.

rv–error correlation (val daily Q90): **0.3675**

## Variant Comparison

| variant         |   rel_score |   q90_abs_e |   q95_abs_e |   daily_violation_gt_035 |   daily_violation_share |   share_abs_e_gt_050 |       DA |
|:----------------|------------:|------------:|------------:|-------------------------:|------------------------:|---------------------:|---------:|
| anchor_baseline |    0.044717 |    0.046947 |    0.062184 |                      364 |                0.552352 |             0.08856  | 0.518273 |
| 1d_regime       |    0.046476 |    0.046831 |    0.062145 |                      362 |                0.549317 |             0.088179 | 0.518273 |
| 2d_regime       |    0.048546 |    0.046649 |    0.062122 |                      361 |                0.5478   |             0.087534 | 0.518273 |

## 1D Per-Decile Scales

|   decile |      lo |      hi |   scale |   train_rs |     n |
|---------:|--------:|--------:|--------:|-----------:|------:|
|        1 | 0.01065 | 0.01544 |   0.95  |    0.0263  | 11468 |
|        2 | 0.01544 | 0.01653 |   0.825 |    0.02853 | 11445 |
|        3 | 0.01653 | 0.01732 |   0.875 |    0.03571 | 11552 |
|        4 | 0.01732 | 0.01811 |   1.025 |    0.03553 | 11486 |
|        5 | 0.01811 | 0.01894 |   0.725 |    0.02783 | 11479 |
|        6 | 0.01894 | 0.01976 |   0.8   |    0.04872 | 11482 |
|        7 | 0.01976 | 0.02078 |   1.35  |    0.05429 | 11512 |
|        8 | 0.02078 | 0.02215 |   0.975 |    0.04637 | 11505 |
|        9 | 0.02215 | 0.02412 |   1.075 |    0.04521 | 11476 |
|       10 | 0.02412 | 0.02969 |   1.125 |    0.11579 | 11533 |

## Conformal Interval Metrics

| variant                   |   train_quantile |   coverage |   mean_width |   median_width |   q90_width |
|:--------------------------|-----------------:|-----------:|-------------:|---------------:|------------:|
| plain_conformal_90pct     |        0.0338507 |   0.827992 |    0.0677013 |      0.0677013 |   0.0677013 |
| hetero_rv_conformal_90pct |        1.72924   |   0.854297 |    0.075009  |      0.0729368 |   0.0937565 |
| hetero_rv_conformal_95pct |        2.34115   |   0.921565 |    0.101552  |      0.0987463 |   0.126933  |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/regime_calibration_20260527",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/regime_calibration_20260527"
}