# Distributional Scale LSTM Train Probe - Mode Summary

Aggregate by selection mode across seeds 43, 52, 71.

| variant | mode | policies | seeds | rel_score | daily p90 | daily max | days >=7% | days >=8% |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| `dist_scale_global` | `stability` | `stability:tail_shrink_s50_m50, stability:tail_shrink_s50_m80` | 3 | 0.01811 | 6.46% | 8.61% | 27.0 | 4.0 |
| `baseline_stressaux_w20` | `raw_or_baseline` | `raw` | 3 | 0.02477 | 6.39% | 9.44% | 29.0 | 7.7 |
| `dist_scale_global` | `raw_or_baseline` | `raw` | 3 | 0.02477 | 6.39% | 9.44% | 29.0 | 7.7 |
| `dist_scale_global` | `balanced` | `balanced:daily_scale_x25_lo50_hi125, balanced:raw` | 3 | 0.02631 | 6.37% | 9.64% | 30.0 | 8.7 |

Read:

- `balanced` does not beat baseline: rel_score improves slightly but max spike is worse.
- `stability` reduces daily max and days >=8%, but it pays with a clear rel_score drop.
- This supports the hypothesis that the sidecar is mostly learning uncertainty/shrinkage, not better tail-direction timing.