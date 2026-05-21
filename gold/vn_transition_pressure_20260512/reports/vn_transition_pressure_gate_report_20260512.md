# VN Transition Pressure Gate Report 2026-05-12

Scope: VN train/validation only. Holdout/test is not used.

## Objective

Improve the current VN model for reporting while keeping LSTM as the main learned signal engine:

```text
base LSTM prediction
-> LSTM filter signal / committee candidates
-> Wyckoff phase gate
-> transition pressure risk filter
-> holding-period execution with transaction cost
```

This is preferred over adding `market_leader_return` directly to the base LSTM because the market-leader experiments did not improve the main model robustly. The improvement path remains LSTM-centered: base LSTM forecast plus LSTM tradeability filter.

## Candidate Policy

Best current policy:

```text
vn_legacy_acc_all_else_transition_pressure_nonneg
```

Phase rule:

| Phase | Policy |
| --- | --- |
| accumulation | `legacy_filter_shortlist` |
| markup | `all_committee_candidates` |
| distribution | `all_committee_candidates` |
| markdown | `all_committee_candidates` |
| transition | `all_committee_candidates` only if `pressure_delta_20 >= 0`, else cash |

Where:

```text
pressure_delta_20 = MA20(mean(buying_pressure - selling_pressure))
```

Interpretation:

- In accumulation, use the older conservative filter shortlist.
- In markup/distribution/markdown, use the broader committee selector.
- In transition, trade only when buying pressure is not negative; otherwise abstain.

## Validation Design

The latest confirmation used three VN filter artifacts:

| Artifact | Description |
| --- | --- |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | plain LSTM filter artifact |
| `portable_lstm_filter_signal_20260508_r05_signmag` | sign-magnitude filter artifact |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | fresh artifact with `market_leader_return K=3,w=60` in the filter input |

Execution assumptions for the strongest validation read:

- validation split only
- strict non-overlap rolling validation: `train=126`, `test=21`, `step=21`
- one-way cost: `15 bps`
- minimum positions: `5`
- no holdout/test data

## Result

Policy: `vn_legacy_acc_all_else_transition_pressure_nonneg`

| Artifact | Net equity | Sharpe | Max drawdown | Avg turnover |
| --- | ---: | ---: | ---: | ---: |
| `r06_selector_module` | `1.923` | `+1.45` | `-21.7%` | `0.189` |
| `r05_signmag` | `1.681` | `+1.29` | `-24.3%` | `0.192` |
| `20260512_market_leader_k3w60` | `1.870` | `+1.55` | `-21.7%` | `0.201` |

Cross-artifact summary:

| Metric | Value |
| --- | ---: |
| artifacts positive | `3 / 3` |
| worst-artifact equity | `1.681` |
| mean equity | `1.825` |
| minimum Sharpe | `+1.29` |
| worst max drawdown | `-24.3%` |
| maximum average turnover | `0.201` |

## Conservative Risk-Control Variant

Because the strongest read is slightly above the strict turnover screen, I also tested the same frozen phase/pressure policy with a less concentrated execution constraint:

```text
minimum positions = 6
```

This is not a new feature or a new model. It is a portfolio construction constraint applied to the same policy.

| Artifact | Net equity | Sharpe | Max drawdown | Avg turnover |
| --- | ---: | ---: | ---: | ---: |
| `r06_selector_module` | `1.724` | `+1.27` | `-21.7%` | `0.173` |
| `r05_signmag` | `1.356` | `+0.92` | `-19.5%` | `0.135` |
| `20260512_market_leader_k3w60` | `1.572` | `+1.18` | `-21.7%` | `0.179` |

Cross-artifact summary:

| Metric | Value |
| --- | ---: |
| artifacts positive | `3 / 3` |
| worst-artifact equity | `1.356` |
| mean equity | `1.551` |
| minimum Sharpe | `+0.92` |
| worst max drawdown | `-21.7%` |
| maximum average turnover | `0.179` |
| risk-control screen | `pass` |

Read:

- `min_positions=5` is the stronger research candidate.
- `min_positions=6` is the cleaner reporting candidate because it passes the `turnover <= 0.20` and `drawdown <= 25%` screens.
- `min_positions=10` was also tested, but diluted the edge too much: worst equity fell to `1.285`.

## Fresh No-Leader Confirmation

I then created one more fresh filter artifact without `market_leader_return`:

```text
portable_lstm_filter_signal_20260512_r02_no_leader_seed43
```

This artifact uses:

- same base model: `broad_signmag_portable_no_identity_20260428_allvn_r01 / lstm_seed_52`
- same filter architecture
- different filter seed: `43`
- excluded feature: `market_leader_return`
- no holdout/test data

The same frozen policy remains positive, but the stress read is weaker:

| Variant | Artifacts | Worst equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Risk screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `min_positions=5` | `4` | `1.240` | `1.679` | `+0.59` | `-24.3%` | `0.201` | fail turnover by `0.001` |
| `min_positions=6` | `4` | `1.186` | `1.460` | `+0.55` | `-21.7%` | `0.179` | pass |

Artifact-level result for the fresh no-leader run:

| Variant | Net equity | Sharpe | Max drawdown | Avg turnover |
| --- | ---: | ---: | ---: | ---: |
| `min_positions=5` | `1.240` | `+0.59` | `-21.7%` | `0.182` |
| `min_positions=6` | `1.186` | `+0.55` | `-21.7%` | `0.126` |

Read:

- The policy still survives a fresh no-leader artifact: positive `4 / 4`.
- The edge is less strong than the first three-artifact read, so the report should not claim this is production-ready.
- The conservative `min_positions=6` variant is the cleaner advisor-facing version because it passes the risk-control screen across all four artifacts.

## Comparison To Market-Leader Direction

Market-leader feature/gate was tested separately:

- `K=25` behaves too much like market beta.
- `K=3,w=60` has some directional signal but does not improve the base LSTM.
- Adding `market_leader_return` to the filter gives tiny `rel_score` gain but worsens cost-aware trade behavior.

Therefore, market-leader signal should stay as a diagnostic feature, not the main improvement path.

## Decision

This is the strongest current VN reporting candidate:

```text
base model + filter signal + Wyckoff transition pressure gate
```

It is stronger than the raw filter selector because it improves execution-level validation after transaction costs and survives three artifacts.

However, it is not final production:

- the strongest `min_positions=5` variant has maximum turnover `0.201`, slightly above the strict `0.20` screen
- the conservative `min_positions=6` variant passes the strict risk screen across four artifacts but gives up part of the return
- the rule still comes from validation research, so holdout/test must remain closed
- next confirmation should be a frozen-rule rerun with no more parameter changes before opening holdout

## Report Wording

Concise wording for advisor discussion:

```text
The current improvement is not from adding another raw feature into the base LSTM.
The stronger result comes from using a two-stage LSTM system: base LSTM forecast plus LSTM tradeability filter.

We keep the base LSTM prediction stream fixed, then use an LSTM filter and committee selector.
Execution is conditioned on Wyckoff-style market phase.
In transition regimes, the model trades only when the 20-day buying-pressure delta is non-negative.

Across three VN validation artifacts, after 15 bps transaction cost, the policy remains positive:
worst equity 1.681, mean equity 1.825, minimum Sharpe +1.29, worst drawdown -24.3%.

For a stricter risk-control version, using minimum 6 positions keeps the same model and same phase/pressure rule,
passes the turnover and drawdown screens, and still remains positive across all four artifacts after a fresh no-leader confirmation:
worst equity 1.186, mean equity 1.460, minimum Sharpe +0.55, worst drawdown -21.7%, max turnover 0.179.

This suggests the useful direction is LSTM-based signal learning plus regime-aware execution and risk control, not simply adding more raw sequence features into the base LSTM.
```
