# Multi-Market Framework Summary

Protocol: train <= 2020-03-31; validation 2020-04-01..2022-11-15. Holdout/test not used.

## LSTM Multi-Market Results

| market   | model                             |   rel_score_mean |   rel_score_std |   DA_mean |   pred_actual_q90_ratio_mean |   n_seeds | note                                         |
|:---------|:----------------------------------|-----------------:|----------------:|----------:|-----------------------------:|----------:|:---------------------------------------------|
| VN       | portable_hetero_combined          |         0.002556 |        0.005571 |  0.47652  |                     0.089989 |         3 | portable common feature LSTM                 |
| VN       | frozen_calibrated_ensemble_anchor |         0.04478  |      nan        |  0.5183   |                   nan        |         5 | current VN validation anchor; holdout closed |
| US       | portable_hetero_combined          |         0.002288 |        0.001041 |  0.51382  |                     0.098347 |         3 | portable common feature LSTM                 |
| US       | context_hetero_combined           |         0.000801 |        0.002389 |  0.517515 |                     0.112682 |         3 | market context appended to LSTM              |
| JP       | portable_hetero_combined          |         0.001016 |        0.002899 |  0.512625 |                     0.084512 |         3 | portable common feature LSTM                 |
| JP       | context_hetero_combined           |         0.002067 |        0.00378  |  0.506142 |                     0.065637 |         3 | market context appended to LSTM              |

## Alpha Sidecar Attempts (US)

| market   | model                    |   raw_rel_score_mean |   alpha_rel_score_mean |   raw_DA_mean |   alpha_DA_mean | note                        |
|:---------|:-------------------------|---------------------:|-----------------------:|--------------:|----------------:|:----------------------------|
| US       | alpha_aux_target         |             0.000237 |              -0.0003   |      0.496333 |        0.509023 | alpha sidecar; not promoted |
| US       | two_head_joint_w005      |            -0.00132  |              -0.003425 |      0.509255 |        0.499011 | alpha sidecar; not promoted |
| US       | two_head_frozen_backbone |            -0.000126 |              -0.002947 |      0.516752 |        0.496881 | alpha sidecar; not promoted |

## Interpretation

- VN anchor remains unchanged: `frozen_calibrated_ensemble_anchor` is still the protected validation candidate (`rel_score=0.04478`).
- JP improved today: adding market/cross-sectional context to portable heteroscedastic LSTM raised JP validation `rel_score` from `0.001016` to `0.002067`.
- US portable raw model remains stronger than context-appended raw LSTM (`0.002288` vs `0.000801`), so context should be treated as a market-specific adapter rather than a universal input.
- Alpha-objective sidecars are not promoted: they improve some alpha diagnostics but hurt or fail to preserve raw `rel_score`.
- Paper framing: portable core + market-specific context adapters, with VN-specific calibrated ensemble/gate as the high-performing market specialization.

## Artifacts

```json
{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_framework_summary_20260526",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/multimarket_framework_summary_20260526"
}
```
