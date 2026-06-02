# VN30 Meta-Ensemble Simulation Report

Date: 2026-05-29
Scope: VN30 constituents during VN validation period only. Holdout/test not used.

Constituents loaded from: `vn30_historical.csv`.
VN30 train observations: 31548; VN30 validation observations: 18329.

## Protocol

- `_broad`: calibration/meta learner trained on the full VN training set, then evaluated only on VN30 validation rows.
- `_vn30`: calibration/meta learner trained only on VN30 training rows, then evaluated on VN30 validation rows.
- Market-regime features remain lagged full-market features; holdout/test is not used.

## Performance Table

| variant             | model     |   alpha |   n_obs |   n_days |   train_rel_score |   rel_score |   q90_abs_e |   q95_abs_e |   share_abs_e_gt_050 |   daily_violation_gt_035 |       DA |
|:--------------------|:----------|--------:|--------:|---------:|------------------:|------------:|------------:|------------:|---------------------:|-------------------------:|---------:|
| hgb_abs_blend_broad | hgb_abs   |   0.475 |   18329 |      659 |          0.041141 |    0.020624 |    0.038804 |    0.052799 |             0.058159 |                      225 | 0.518032 |
| anchor              | anchor    |   0     |   18329 |      659 |          0.038604 |    0.019347 |    0.038747 |    0.052968 |             0.05805  |                      221 | 0.518032 |
| et_tail_blend_broad | et_tail   |   0.675 |   18329 |      659 |          0.039252 |    0.019296 |    0.038756 |    0.052632 |             0.056959 |                      219 | 0.517049 |
| enet_blend_broad    | enet      |   0.375 |   18329 |      659 |          0.04194  |    0.018841 |    0.038755 |    0.052766 |             0.057232 |                      219 | 0.516995 |
| ridge_blend_broad   | ridge     |   0.325 |   18329 |      659 |          0.04266  |    0.017891 |    0.038809 |    0.052798 |             0.057286 |                      217 | 0.516886 |
| hgb_abs_blend_vn30  | hgb_abs   |   0.425 |   18329 |      659 |          0.046767 |    0.017029 |    0.038835 |    0.052692 |             0.05685  |                      220 | 0.51825  |
| 2d_regime_broad     | 2d_regime |   0     |   18329 |      659 |          0.043132 |    0.01682  |    0.038855 |    0.052778 |             0.057559 |                      217 | 0.518032 |
| et_tail_blend_vn30  | et_tail   |   0.275 |   18329 |      659 |          0.040943 |    0.016331 |    0.038841 |    0.052674 |             0.057123 |                      219 | 0.516722 |
| ridge_blend_vn30    | ridge     |   0.3   |   18329 |      659 |          0.042522 |    0.015166 |    0.038862 |    0.05302  |             0.057505 |                      220 | 0.518413 |
| 2d_regime_vn30      | 2d_regime |   0     |   18329 |      659 |          0.03959  |    0.015071 |    0.038846 |    0.052943 |             0.057723 |                      218 | 0.518032 |
| enet_blend_vn30     | enet      |   0.4   |   18329 |      659 |          0.042865 |    0.01488  |    0.038846 |    0.05292  |             0.057505 |                      221 | 0.517541 |

## Interpretation

- Best broad-trained variant: `hgb_abs_blend_broad`.
- Best VN30-trained variant: `hgb_abs_blend_vn30`.
- If broad-trained variants win, the calibration is benefiting from larger cross-sectional training support.
- If VN30-trained variants win, the VN30 universe has enough distinct structure to justify a dedicated calibration layer.

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/vn30_meta_ensemble_simulation_20260529",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/vn30_meta_ensemble_simulation_20260529"
}