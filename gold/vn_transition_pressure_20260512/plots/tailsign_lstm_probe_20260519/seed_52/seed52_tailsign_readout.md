# Tailsign / Tailstress Probe - Seed 52

Validation only. Tests direct tail features and directional penalty variants.

| variant | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05_tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 | 0.505 |
| `plain_global_weighted_mild_tail35_sign_p10` | 0.02827 | 4.825% | 6.342% | 10.793% | 33 | 12 | 0.169 | 0.504 |
| `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 | 0.504 |
| `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 | 0.504 |
| `plain_global_weighted_mild_tail35_sign_p05` | 0.02695 | 4.808% | 6.469% | 12.353% | 40 | 16 | 0.193 | 0.505 |
| `plain_global_weighted_mild_tail35_p05_tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 | 0.141 | 0.498 |
| `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 | 0.496 |
| `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 | 0.493 |

## Read

- Current-day tailstress gives the highest rel_score but worsens max spike; direct feature injection is too unconstrained.
- Strict-past tailstress improves spike count versus weighted mild but loses rel_score versus tail-loss only.
- Directional penalty variants do not improve spike control; they are not worth promoting now.
- Best balanced candidate remains `plain_global_weighted_mild_tail35_p05`.