# Regime-Conditional Calibration

Protocol: lagged-rv regime calibration learned on train; validation readout only. Holdout/test not used.

**rv-error correlation (val)**: 0.4199

## Comparison: Baseline vs Regime-Calibrated

| variant           |   rel_score |   q90_abs_e |   q95_abs_e |   daily_violation_gt_035 |   daily_violation_share |   share_abs_e_gt_050 |       DA |
|:------------------|------------:|------------:|------------:|-------------------------:|------------------------:|---------------------:|---------:|
| anchor_baseline   |    0.044717 |    0.046947 |    0.062184 |                      364 |                0.552352 |             0.08856  | 0.518273 |
| regime_calibrated |    0.047687 |    0.046697 |    0.062069 |                      359 |                0.544765 |             0.087484 | 0.518273 |

## Per-Decile Scales (learned on train)

|   decile |      lo |      hi |   scale |   train_rs |     n |
|---------:|--------:|--------:|--------:|-----------:|------:|
|        1 | 0.01078 | 0.01495 |   0.9   |    0.01742 | 11440 |
|        2 | 0.01495 | 0.01592 |   0.9   |    0.03276 | 11487 |
|        3 | 0.01592 | 0.01666 |   0.9   |    0.04788 | 11473 |
|        4 | 0.01666 | 0.0174  |   0.75  |    0.03466 | 11499 |
|        5 | 0.0174  | 0.01819 |   0.775 |    0.03592 | 11547 |
|        6 | 0.01819 | 0.019   |   0.85  |    0.04208 | 11455 |
|        7 | 0.019   | 0.02006 |   1.225 |    0.04775 | 11497 |
|        8 | 0.02006 | 0.0212  |   1.075 |    0.05448 | 11522 |
|        9 | 0.0212  | 0.02314 |   1.175 |    0.06393 | 11496 |
|       10 | 0.02314 | 0.0315  |   1.225 |    0.09424 | 11522 |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/regime_calibration_20260527",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/regime_calibration_20260527"
}