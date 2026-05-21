# Tailstress Gate Decision - Seed 52

Best candidate by rel_score: `conservative_mean_positive_gap`.

| candidate | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|
| `base` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 |
| `tail_loss` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 |
| `tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 |
| `tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 |
| `simplex_rel` | 0.03198 | 4.807% | 6.289% | 11.152% | 33 | 10 |
| `simplex_tail` | 0.03569 | 4.782% | 6.300% | 11.770% | 30 | 13 |
| `conservative_mean_positive_gap` | 0.03590 | 4.767% | 6.259% | 9.957% | 21 | 11 |
| `conservative_tailstress_abs` | 0.03132 | 4.804% | 6.278% | 10.516% | 24 | 9 |
| `instance` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 |

## Read

- Tailstress is useful as a regime/confidence signal, but direct use still amplifies tail spikes.
- Conservative gate improves `rel_score` and reduces `>=7%` spike days versus `tail_loss`.
- It does not beat `tail_loss` on `>=8%` spike days, so this is not final.
- Next valid step is multi-seed tailstress-gate validation, not freezing a new gold model.