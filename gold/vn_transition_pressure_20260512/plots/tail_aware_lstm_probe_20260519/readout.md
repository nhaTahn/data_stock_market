# Tail-Aware LSTM Probe Readout

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether ideas from imbalanced regression, probabilistic forecasting, and tail-aware auxiliary supervision fit the current LSTM setup.

## Tested Variants

All variants keep the same LSTM backbone `[64, 32]`, window `15`, seed `52`, and the 29-feature portable gold feature set.

| variant | idea tested |
|:--|:--|
| `plain_global_rel` | current-style global train z-score + rel_score loss |
| `plain_global_weighted_mild` | current-style input + mild weighting for larger absolute returns |
| `plain_global_weighted` | current-style input + weighted rel_score for large absolute returns |
| `plain_global_instance_rel` | current global scaling plus per-window instance z-score |
| `plain_global_instance_weighted_mild` | instance z-score + mild large-return weighting |
| `plain_multimarket_rel` | strict-past rolling z-score + cross-sectional z/rank normalization |
| `plain_multimarket_weighted_mild` | multimarket normalization + mild large-return weighting |
| `tailaware_multimarket_weighted` | multimarket normalization + weighted rel_score + auxiliary tail/magnitude heads |

## Validation Result

| variant | rel_score | pooled q90 error | daily q90 p90 | daily q90 max | days >= 7% | days >= 8% | prediction/actual abs q90 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_rel` | 0.01016 | 4.971% | 6.568% | 8.481% | 45 | 12 | 0.115 |
| `plain_global_weighted_mild` | 0.01099 | 4.933% | 6.585% | 9.814% | 38 | 9 | 0.112 |
| `plain_global_weighted` | -0.00848 | 4.745% | 6.277% | 12.443% | 38 | 18 | 0.343 |
| `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |
| `plain_global_instance_weighted_mild` | 0.00864 | 4.957% | 6.593% | 8.825% | 40 | 3 | 0.127 |
| `plain_multimarket_rel` | 0.00172 | 5.065% | 6.625% | 7.467% | 36 | 0 | 0.081 |
| `plain_multimarket_weighted_mild` | -0.00022 | 5.059% | 6.553% | 8.059% | 24 | 1 | 0.121 |
| `tailaware_multimarket_weighted` | -0.05075 | 4.974% | 6.524% | 10.734% | 53 | 19 | 0.397 |

## Target Status

Primary target status on validation:

| target | best/current status | pass? |
|:--|:--|:--|
| keep positive `rel_score` | best is `plain_global_weighted_mild` at `0.01099`, slightly above baseline `0.01016` | yes |
| pooled q90 abs error below `5%` | `plain_global_weighted_mild` is `4.933%`; baseline is `4.971%` | yes |
| reduce `>=7%` daily spike days | `plain_multimarket_weighted_mild` has 24 days, but rel_score is slightly negative; best positive-rel variant is `plain_global_weighted_mild` with 38 days vs baseline 45 | partial |
| remove/reduce `>=8%` daily spike days | `plain_multimarket_rel` has 0 days but weak rel_score; best positive-rel variant is `plain_global_instance_rel` with 1 day but lower rel_score | partial |
| daily q90 p90 below `5%` | best tested value is `6.277%` from aggressive weighting, but that variant has negative rel_score and worse max spikes | no |
| daily q90 max below `7%` | no all-days forecasting variant passes; best is `plain_multimarket_rel` at `7.467%` | no |

## Interpretation

The useful result is about mild imbalance treatment and input processing, not the current heavy tail-aware head.

1. `plain_global_weighted_mild` is the best rel_score variant in this quick probe and also improves pooled q90 error.
2. `plain_global_weighted_mild` reduces `>=7%` and `>=8%` spike counts versus baseline, but the maximum daily q90 error worsens. It is a candidate, not a final answer.
3. `plain_global_instance_rel` and `plain_multimarket_rel` are useful diagnostic variants: they suppress the worst `>=8%` spikes, but they lose rel_score and under-amplify predictions.
4. `plain_global_weighted` is too aggressive. It reduces pooled q90 error, but creates larger max spikes and negative rel_score.
5. `tailaware_multimarket_weighted` is not acceptable as configured: it increases prediction amplitude too much, makes rel_score negative, and worsens severe spikes.

## Research Read

The model is not simply lacking amplitude. The baseline prediction abs q90 is only about `11.5%` of actual abs q90, but naively increasing amplitude hurts if direction/timing is not right. The weighted and tail-aware variants increase amplitude, but both make rel_score worse and create worse maximum spikes.

The more promising direction is:

```text
Keep point forecast LSTM simple.
Use mild target imbalance treatment.
Use normalization as a stability diagnostic, not yet as the final default.
Do not add a heavy tail auxiliary head until it is constrained/calibrated.
```

## Next Hypothesis

The next test should be narrower:

1. Keep `plain_global_rel` as the anchor.
2. Add only a mild instance-normalization or hybrid-normalization branch.
3. If using imbalance weighting, make it milder than the current weighted loss.
4. Train across multiple seeds before claiming improvement.

Suggested next variants:

| next variant | reason |
|:--|:--|
| `plain_global_weighted_mild` long run | current best balance of positive rel_score and pooled q90 error |
| `plain_global_rel` long run | baseline anchor; do not replace it until multi-seed validation passes |
| `plain_global_instance_rel` as stability challenger | almost removes `>=8%` spikes, but needs amplitude recovery |
| `plain_global_rel + mild_LDS_weights` | directly tests deep imbalanced regression with smoother weights than the aggressive variant |
| `plain_global_rel + two-stage train` | first rel_score, then short low-lr tail-weighted fine-tune |

Current conclusion: the `<5%` pooled q90 target is met by `plain_global_weighted_mild`, and rel_score improves slightly. The stricter daily-spike target is not met yet. The next improvement should be a longer, multi-seed run of the mild recipe plus a controlled amplitude/stability check, not a larger architecture.
