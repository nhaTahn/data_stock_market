# Meta-Ensemble Calibration

Protocol: train-only meta learners over frozen 5-seed predictions + lagged market regime. Validation readout only. Holdout/test not used.

## Comparison

| variant           | model       |   alpha |   train_rel_score |   rel_score |   q90_abs_e |   q95_abs_e |   share_abs_e_gt_050 |   daily_violation_gt_035 |       DA |
|:------------------|:------------|--------:|------------------:|------------:|------------:|------------:|---------------------:|-------------------------:|---------:|
| hgb_abs_blend     | hgb_abs     |   0.475 |          0.044339 |    0.04881  |    0.046724 |    0.062444 |             0.087964 |                      360 | 0.518273 |
| 2d_regime         | 2d_regime   |   0     |        nan        |    0.048546 |    0.046649 |    0.062122 |             0.087534 |                      361 | 0.518273 |
| et_tail_blend     | et_tail     |   0.675 |          0.042423 |    0.047876 |    0.046592 |    0.062215 |             0.086823 |                      360 | 0.517512 |
| ridge_blend       | ridge       |   0.325 |          0.041993 |    0.047077 |    0.046719 |    0.062028 |             0.087203 |                      362 | 0.516982 |
| enet_blend        | enet        |   0.375 |          0.042138 |    0.046993 |    0.046717 |    0.062008 |             0.087451 |                      360 | 0.516784 |
| anchor            | anchor      |   0     |        nan        |    0.044717 |    0.046947 |    0.062184 |             0.08856  |                      364 | 0.518273 |
| ridge_poly2_blend | ridge_poly2 |   0.75  |          0.043257 |    0.043304 |    0.046933 |    0.061864 |             0.088196 |                      359 | 0.515775 |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_calibration_20260528",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/meta_ensemble_calibration_20260528"
}