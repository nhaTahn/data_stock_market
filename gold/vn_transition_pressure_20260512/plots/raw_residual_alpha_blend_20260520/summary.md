# Raw + Residual-Alpha Blend

Scope: reuse existing 3-seed predictions from residual target probe. Blend weight is selected on late-train only, then evaluated on validation.

| mode | policies | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% | days >=8% | pred abs q90 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `selected_blend` | `selected_blend_w_0.1, selected_blend_w_0.5, selected_blend_w_0.6` | 3 | 0.02938 | 4.83% | 3.69% | 6.31% | 10.88% | 159.0 | 32.7 | 11.3 | 0.83% |
| `raw_baseline` | `raw_baseline` | 3 | 0.02837 | 4.82% | 3.72% | 6.32% | 11.12% | 159.0 | 36.0 | 11.7 | 0.92% |
| `alpha_only_no_market` | `alpha_only_no_market` | 3 | 0.02574 | 4.86% | 3.71% | 6.36% | 10.80% | 165.3 | 31.7 | 10.3 | 0.81% |
| `existing_residual_lagged_ar1` | `existing_residual_lagged_ar1` | 3 | 0.02436 | 4.85% | 3.70% | 6.39% | 10.68% | 160.3 | 33.3 | 10.0 | 0.83% |

## Read

- `alpha_only_no_market` tests residual target without adding any market component back.
- `selected_blend` tests whether raw-return prediction and residual-alpha prediction are complementary under a late-train selection rule.
- If selected blend fails to beat `raw_baseline`, residual target is not yet a practical input/target processing improvement for full-coverage next-day return.