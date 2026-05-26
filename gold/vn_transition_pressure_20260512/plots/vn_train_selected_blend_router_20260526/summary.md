# VN Train-Selected Blend Router

Protocol: train <= 2020-03-31 for selection; validation 2020-04-01..2022-11-15 for readout. Holdout/test not used.

## Decision

- Selected variant: `old_anchor`
- Promote blend: `False`
- Selected validation rel_score: `0.04478`
- Old anchor validation rel_score: `0.04478`

## Top Train-Selected Candidates

| variant         |   blend_weight_rich |   train_objective |   train_mean_rel |   train_std_rel |   train_min_rel |   train_positive_folds |   val_rel_score |   val_DA |
|:----------------|--------------------:|------------------:|-----------------:|----------------:|----------------:|-----------------------:|----------------:|---------:|
| blend_rich_0.00 |                0    |          0.010233 |         0.040553 |        0.045616 |       -0.018252 |                     77 |        0.044775 | 0.518273 |
| blend_rich_0.15 |                0.15 |          0.009759 |         0.041282 |        0.045255 |       -0.016968 |                     80 |        0.044693 | 0.519894 |
| blend_rich_0.10 |                0.1  |          0.009497 |         0.041124 |        0.045784 |       -0.018333 |                     78 |        0.043678 | 0.519315 |
| blend_rich_0.20 |                0.2  |          0.009492 |         0.041621 |        0.045029 |       -0.016388 |                     80 |        0.04461  | 0.51996  |
| blend_rich_0.05 |                0.05 |          0.009245 |         0.040688 |        0.045683 |       -0.017862 |                     77 |        0.044436 | 0.518554 |
| blend_rich_0.25 |                0.25 |          0.008953 |         0.041548 |        0.045549 |       -0.016299 |                     80 |        0.044744 | 0.51953  |
| blend_rich_0.30 |                0.3  |          0.00893  |         0.041337 |        0.045338 |       -0.016393 |                     81 |        0.04422  | 0.519514 |
| blend_rich_0.35 |                0.35 |          0.008111 |         0.041321 |        0.045488 |       -0.016557 |                     81 |        0.044292 | 0.519398 |
| blend_rich_0.50 |                0.5  |          0.007498 |         0.040946 |        0.045138 |       -0.018024 |                     78 |        0.043099 | 0.519745 |
| blend_rich_0.45 |                0.45 |          0.007406 |         0.041018 |        0.044945 |       -0.016967 |                     79 |        0.043564 | 0.519282 |
| blend_rich_0.40 |                0.4  |          0.006961 |         0.040972 |        0.045171 |       -0.017841 |                     79 |        0.043904 | 0.519464 |
| blend_rich_0.55 |                0.55 |          0.00606  |         0.040861 |        0.044776 |       -0.018011 |                     79 |        0.042943 | 0.519911 |

## Top Validation Readout (diagnostic, not selection)

|     n |   rel_score |   absE_robust |   base_robust |   absE_q90 |       DA |   pred_actual_q90_ratio | variant         |   blend_weight_rich | selected_by_train   |
|------:|------------:|--------------:|--------------:|-----------:|---------:|------------------------:|:----------------|--------------------:|:--------------------|
| 60445 |    0.044775 |      0.036003 |      0.037691 |   0.046932 | 0.518273 |                0.193205 | blend_rich_0.00 |                0    | False               |
| 60445 |    0.044775 |      0.036003 |      0.037691 |   0.046932 | 0.518273 |                0.193205 | old_anchor      |                0    | True                |
| 60445 |    0.044744 |      0.036004 |      0.037691 |   0.047017 | 0.51953  |                0.189893 | blend_rich_0.25 |                0.25 | False               |
| 60445 |    0.044693 |      0.036006 |      0.037691 |   0.046993 | 0.519894 |                0.191132 | blend_rich_0.15 |                0.15 | False               |
| 60445 |    0.04461  |      0.036009 |      0.037691 |   0.047013 | 0.51996  |                0.1903   | blend_rich_0.20 |                0.2  | False               |
| 60445 |    0.044436 |      0.036016 |      0.037691 |   0.04697  | 0.518554 |                0.19233  | blend_rich_0.05 |                0.05 | False               |
| 60445 |    0.044292 |      0.036021 |      0.037691 |   0.047055 | 0.519398 |                0.190658 | blend_rich_0.35 |                0.35 | False               |
| 60445 |    0.04422  |      0.036024 |      0.037691 |   0.047042 | 0.519514 |                0.190171 | blend_rich_0.30 |                0.3  | False               |
| 60445 |    0.043904 |      0.036036 |      0.037691 |   0.047081 | 0.519464 |                0.191264 | blend_rich_0.40 |                0.4  | False               |
| 60445 |    0.043678 |      0.036045 |      0.037691 |   0.047039 | 0.519315 |                0.191526 | blend_rich_0.10 |                0.1  | False               |
| 60445 |    0.043564 |      0.036049 |      0.037691 |   0.047086 | 0.519282 |                0.191971 | blend_rich_0.45 |                0.45 | False               |
| 60445 |    0.043099 |      0.036066 |      0.037691 |   0.047108 | 0.519745 |                0.192473 | blend_rich_0.50 |                0.5  | False               |

## Payload

```json
{
  "selected_variant": "old_anchor",
  "promote_blend": false,
  "selected_train_candidate": {
    "variant": "blend_rich_0.00",
    "blend_weight_rich": 0.0,
    "train_objective": 0.010232550682008592,
    "train_mean_rel": 0.04055269923262708,
    "train_std_rel": 0.04561631222773813,
    "train_min_rel": -0.018252200060985047,
    "train_positive_folds": 77,
    "val_rel_score": 0.04477518130782343,
    "val_DA": 0.5182728099925552
  },
  "selected_validation_metric": {
    "n": 60445,
    "rel_score": 0.04477516686817884,
    "absE_robust": 0.03600317947566509,
    "base_robust": 0.037690790928900236,
    "absE_q90": 0.046931590884923935,
    "DA": 0.5182728099925552,
    "pred_actual_q90_ratio": 0.19320456683635712,
    "variant": "old_anchor",
    "blend_weight_rich": 0.0,
    "selected_by_train": true
  },
  "old_anchor_validation_rel_score": 0.04477516686817884,
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/vn_train_selected_blend_router_20260526",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/vn_train_selected_blend_router_20260526"
}
```