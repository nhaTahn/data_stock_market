# Tailstress LSTM Probe - Seed 52

Validation only. Tailstress features test whether market breadth/tail context improves the LSTM input processing.

| variant | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05_tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 |
| `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 |
| `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 |
| `plain_global_weighted_mild_tail35_p05_tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 | 0.141 |
| `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |

## Read

- Current-day tailstress has the best `rel_score`, but worsens severe spikes and max daily q90 error.
- Strict-past tailstress is cleaner from a look-ahead perspective; use it only if it beats the no-tailstress candidate on both rel_score and spike control.
- If strict-past tailstress does not help, the breadth feature has signal but needs a constrained/calibrated usage rather than direct feature injection.