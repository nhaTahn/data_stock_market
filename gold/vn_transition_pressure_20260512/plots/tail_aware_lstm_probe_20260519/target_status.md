# Target Status - Tail-Aware LSTM Probe

Scope: validation only, VN train/validation split, seed `52`, 6-epoch probe.

## Result

The current probe partially meets the target.

| target | result | status |
|:--|:--|:--|
| improve `rel_score` versus baseline | `plain_global_weighted_mild`: `0.01099` vs `plain_global_rel`: `0.01016` | pass |
| pooled q90 abs error `<5%` | `plain_global_weighted_mild`: `4.933%` | pass |
| reduce `>=7%` daily spike days | `plain_global_weighted_mild`: 38 vs baseline 45 | partial |
| reduce `>=8%` daily spike days | `plain_global_weighted_mild`: 9 vs baseline 12 | partial |
| daily q90 p90 `<5%` | best tested all-days value remains above `6%` | fail |
| daily q90 max `<7%` | best tested all-days model is `7.467%`; current best rel_score model is `9.814%` | fail |

## Best Candidate

`plain_global_weighted_mild` is the best current candidate if the main goal is improving the LSTM forecasting metric:

- `rel_score`: `0.01099`, better than baseline `0.01016`
- pooled q90 abs error: `4.933%`, better than baseline `4.971%`
- `>=7%` spike days: 38, better than baseline 45
- `>=8%` spike days: 9, better than baseline 12

It is not yet a final model because its max daily q90 error is worse than baseline (`9.814%` vs `8.481%`).

## Decision

Do not freeze this as a new gold model yet.

Use this result as evidence that mild imbalance-aware training has signal. The next run should focus on `plain_global_rel`, `plain_global_weighted_mild`, and one stability challenger such as `plain_global_instance_rel`, across more epochs and multiple seeds.
