# Target/Input Processing Readout 2026-05-20

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether target/input processing can reduce full-coverage next-day return error spikes while keeping LSTM as the main forecaster.

## 1. Residual-alpha target as a practical signal

Probe: reuse existing raw LSTM and residual-target LSTM predictions, then select a blend weight on late-train only.

| mode | seeds | rel_score | daily q90 p90 | daily q90 max | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|
| raw_baseline | 3 | 0.02837 | 6.32% | 11.12% | 36.0 | 11.7 |
| alpha_only_no_market | 3 | 0.02574 | 6.36% | 10.80% | 31.7 | 10.3 |
| selected_blend | 3 | 0.02938 | 6.31% | 10.88% | 32.7 | 11.3 |

Read:

- Residual-alpha contains a small complementary signal: selected blend improves rel_score by about +0.001 versus raw baseline.
- It also lowers daily max and days >=7%, but the effect is small.
- This is not enough to solve spike days. It is a candidate for a light multitask auxiliary target, not a replacement for the current LSTM base.

Artifacts:

- `gold/vn_transition_pressure_20260512/plots/raw_residual_alpha_blend_20260520/summary.md`

## 2. Target scale processing

Probe: train seed 52 stressaux LSTM with alternative target normalization scale:

- baseline: stock `volatility_20`, q25 floor
- floor50: stock `volatility_20`, q50 floor
- mktmaxscale: max(stock volatility, lagged market q90 absolute return EWM5)
- mktblendscale: 70% stock volatility + 30% lagged market q90 absolute return EWM5

| variant | rel_score | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual q90 |
|:--|--:|--:|--:|--:|--:|--:|
| stressaux_w20 baseline | 0.03300 | 6.37% | 9.62% | 29 | 12 | 0.171 |
| floor50 | 0.02136 | 6.47% | 10.79% | 29 | 10 | 0.131 |
| mktmaxscale | 0.02178 | 6.64% | 10.29% | 37 | 12 | 0.122 |
| mktblendscale | 0.02750 | 6.50% | 10.37% | 39 | 15 | 0.172 |

Read:

- Current stock-vol target scale remains better than the tested alternatives.
- Raising the target-scale floor reduces prediction amplitude and hurts direction/rel_score.
- Adding lagged market-tail volatility as target scale does not make the model adapt to future spike days; it mostly changes amplitude and worsens daily p90/max.
- Conclusion: current spike issue is not solved by simple target-scale normalization.

Artifacts:

- `gold/vn_transition_pressure_20260512/plots/target_scale_lstm_probe_20260520/seed_52_screen/summary.md`

## Decision

Do not promote the market-tail target-scale variants.

Promote only the insight:

```text
Residual-alpha target has weak but real complementary signal.
Market-tail target scaling is not enough; spike days require uncertainty/regime/error-control or stronger market-direction information.
```

For the next model iteration, the clean option is:

```text
LSTM return head = keep stock-vol normalized target.
Auxiliary residual-alpha head = optional small regularizer.
Evaluation = full-coverage rel_score plus calibrated daily error stability.
```
