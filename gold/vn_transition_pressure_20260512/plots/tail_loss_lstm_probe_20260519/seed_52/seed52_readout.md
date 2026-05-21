# Tail-Loss LSTM Probe - Seed 52

Validation only. Baseline rows are from the existing multiseed run; tail-loss rows are newly trained.

| variant | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 |
| `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 |
| `plain_global_weighted_mild_tail50_p10` | 0.02628 | 4.826% | 6.386% | 10.814% | 35 | 14 | 0.186 |
| `plain_global_weighted_mild_tail50_p20` | 0.02195 | 4.861% | 6.381% | 10.635% | 37 | 9 | 0.143 |
| `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |

## Read

- `plain_global_weighted_mild_tail35_p05` is the best new candidate on seed 52.
- It keeps rel_score near weighted mild while reducing `>=7%` spike days from 41 to 25 and `>=8%` spike days from 17 to 9.
- Daily max is still above target, so this is a candidate for multi-seed validation, not a final model.