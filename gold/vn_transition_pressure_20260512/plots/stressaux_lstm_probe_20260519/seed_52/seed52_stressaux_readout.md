# StressAux LSTM Probe - Seed 52

Validation only. StressAux keeps the base feature set and adds an auxiliary market-stress head to regularize LSTM hidden state.

| variant | rel_score | q90 error | daily q90 p90 | daily max | days >=7% | days >=8% | pred/actual q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05_tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 | 0.505 |
| `plain_global_weighted_mild_tail35_stressaux_w20` | 0.03300 | 4.796% | 6.372% | 9.624% | 29 | 12 | 0.171 | 0.510 |
| `plain_global_weighted_mild_tail35_stressaux` | 0.03051 | 4.810% | 6.405% | 10.336% | 34 | 14 | 0.169 | 0.503 |
| `plain_global_weighted_mild_tail35_stressaux_past` | 0.02835 | 4.820% | 6.400% | 10.667% | 32 | 9 | 0.178 | 0.505 |
| `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 | 0.504 |
| `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 | 0.504 |
| `plain_global_weighted_mild_tail35_p05_tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 | 0.141 | 0.498 |
| `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 | 0.496 |
| `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 | 0.493 |

## Read

- StressAux improves rel_score versus tail_loss on seed 52.
- `stressaux_w20` is the best StressAux setting: strong rel_score and lower daily max than tail_loss.
- It still has more `>=8%` spike days than tail_loss, so it is not final.
- Run multi-seed only for `stressaux_w20` if we want to validate this branch.