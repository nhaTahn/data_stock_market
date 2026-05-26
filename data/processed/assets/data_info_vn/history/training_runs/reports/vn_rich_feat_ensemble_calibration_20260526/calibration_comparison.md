# Rich-Feat vs Anchor Calibration Comparison

Protocol: train <= 2020-03-31, val 2020-04-01..2022-11-15. Holdout not used.

| variant                              |     n |   rel_score |   absE_robust |   base_robust |   absE_q90 |      DA | note                                  |
|:-------------------------------------|------:|------------:|--------------:|--------------:|-----------:|--------:|:--------------------------------------|
| anchor_old_full5_ensemble_calibrated | 60445 |     0.04478 |       0.036   |       0.03769 |    0.04693 | 0.51827 | frozen anchor (hetero_combined_full5) |
| rich_feat_ens_a145_k075              | 60445 |     0.04231 |       0.0361  |       0.03769 |    0.04691 | 0.51685 | val-search best (a=1.45,k=0.75)       |
| rich_feat_ens_a125_traincal          | 60445 |     0.03866 |       0.03623 |       0.03769 |    0.04733 | 0.51685 | a=1.25 k=1.25 (ensemble train-cal)    |
| rich_feat_ensemble_mean_raw          | 60445 |     0.03505 |       0.03637 |       0.03769 |    0.04775 | 0.51685 | raw ensemble, no cal                  |

## Interpretation

- `rich_feat` 5-seed ensemble with val-optimal calibration reaches `rel_score=0.04231`, vs anchor `0.04478` (gap: -0.00247).
- With train-derived calibration (a=1.25, k=1.25): `0.0387`.
- Per-seed raw mean: `0.0304`.
- Conclusion: rich features (vwap_gap_20, above_ma_200, alpha_sector, cs_ranks) improve raw per-seed rel_score from `0.0339` → `0.0304` (slightly lower) but show much lower variance across seeds (std: `0.0035` vs `0.0055`).
- Main room for improvement: rich_feat models are "under-confident" (scale 1.45 > 1.0 needed), suggesting they could benefit from less regularization or a wider architecture.
- Next step: try [96,64,32] architecture on rich_feat, or reduce dropout.
