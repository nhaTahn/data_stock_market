# Distributional Scale LSTM Train Probe

Scope: train new LSTM heads on VN train, tune output policy on late-train, evaluate on validation. Holdout/test is not used.

- seeds: `43,52,71`
- epochs: `5`
- variants: `dist_scale_global`

| variant | policy | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `dist_scale_global` | `balanced:daily_scale_x25_lo50_hi125` | 1 | 0.02427 | 4.85% | 3.72% | 6.31% | 9.86% | 165.0 | 31.0 |
| `dist_scale_global` | `raw` | 3 | 0.02477 | 4.86% | 3.69% | 6.39% | 9.44% | 166.7 | 29.0 |
| `baseline_stressaux_w20` | `raw` | 3 | 0.02477 | 4.86% | 3.69% | 6.39% | 9.44% | 166.7 | 29.0 |
| `dist_scale_global` | `balanced:raw` | 2 | 0.02734 | 4.84% | 3.69% | 6.40% | 9.52% | 166.5 | 29.5 |
| `dist_scale_global` | `stability:tail_shrink_s50_m50` | 2 | 0.01723 | 4.93% | 3.69% | 6.46% | 8.26% | 169.5 | 26.5 |
| `dist_scale_global` | `stability:tail_shrink_s50_m80` | 1 | 0.01986 | 4.90% | 3.70% | 6.46% | 9.32% | 168.0 | 28.0 |

## Read

- A trained distributional head is useful only if it improves both rel_score and daily q90 spike metrics versus `baseline_stressaux_w20`.
- If rel_score improves but max spike rises, the head is learning residual direction but not stress-day uncertainty.
- If max spike falls but rel_score drops, the head is mostly shrinking forecasts rather than learning better timing.