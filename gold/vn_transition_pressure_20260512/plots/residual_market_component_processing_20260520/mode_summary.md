# Residual Market Component Mode Summary

| mode | policies | seeds | rel_score | daily p90 | daily max | days >=7% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|
| `raw_baseline` | `raw_baseline` | 3 | 0.02837 | 6.32% | 11.12% | 36.0 | 11.7 |
| `existing_residual_lagged_ar1` | `existing_residual_lagged_ar1` | 3 | 0.02436 | 6.39% | 10.68% | 33.3 | 10.0 |
| `selected_lagged_market` | `selected:ewm3_scale_0.1, selected:ewm3_scale_0.25, selected:ewm5_scale_0.25` | 3 | 0.02175 | 6.41% | 10.30% | 31.3 | 10.0 |