# Market Leader Signal Findings 2026-05-12

Scope: VN train/validation only. Holdout/test is not used.

## Hypothesis

Replace the old hard-coded `vingroup_momentum` idea with a causal market-leader basket:

```text
liquidity_score[j,t] = MA_window(traded_value[j,t-1])
leader_set[t] = top K names by liquidity_score[j,t]
market_leader_return[t] = sum_j w[j,t] * return[j,t]
w[j,t] = liquidity_score[j,t] / sum_j liquidity_score[j,t]
```

`vingroup_momentum` is now only a deprecated compatibility alias for `market_leader_return`.

## Standalone Signal Read

Train-selected signal grid:

- selected config: `K=3`, liquidity window `60`, raw leader return
- train rel improvement: `+0.01037`
- validation rel improvement: `+0.00144`
- validation Spearman IC: `+0.0183`
- validation sign accuracy: `54.17%`
- latest selected leaders on `2026-03-31`: `HPG`, `FPT`, `SSI`

Large-K diagnostic:

- `K=25,w=90` has better validation standalone improvement, but its correlation with the equal-weight market proxy is about `0.957`
- treat `K=25` as a market proxy diagnostic, not a distinct leader signal

## LSTM Feature Smoke

Single-seed 12-epoch smoke on the same broad no-identity setup:

| Run | Model | Val rel_score | Val directional |
| --- | ---: | ---: | ---: |
| no leader | LSTM | `0.00675` | `47.57%` |
| no leader | signmag | `0.00724` | `48.54%` |
| `K=3,w=60` | LSTM | `0.00457` | `48.52%` |
| `K=3,w=60` | signmag | `0.00478` | `47.97%` |
| `K=25,w=90` | LSTM | `0.00421` | `48.37%` |
| `K=25,w=90` | signmag | `0.00496` | `47.82%` |

Read: leader return has some directional information, but adding it directly to the LSTM input reduced the main validation `rel_score`.

## Filter/Gate Read

Post-model leader gates on existing filter artifacts:

- direct leader-return agree/riskoff gates reduced `rel_score`
- leader breadth gates reduced `rel_score`
- `prediction_gate_breadth_strong_only` had validation net equity above `1.0` on two artifacts, but coverage was only about `0.7%` to `1.6%` and IC was unstable

Read: do not promote hand-built leader gates.

## Filter-LSTM Input Read

New filter artifact:

- artifact: `reports/filter_signal/portable_lstm_filter_signal_20260512_r01_market_leader_k3w60`
- added filter input: `market_leader_return` with `K=3,w=60`

Comparison versus previous `r06` filter artifact:

| Candidate | Old val rel_score | New val rel_score | Old quartile equity | New quartile equity |
| --- | ---: | ---: | ---: | ---: |
| `move_top_train_ic_selected` | `0.007174` | `0.007292` | `2.142` | `2.030` |
| `gate` | `0.006757` | `0.005574` | `1.078` | `0.906` |
| `base` | `0.004854` | `0.004854` | `2.084` | `2.084` |

After 15 bps cost and train-selected rebalance:

| Candidate | Old best val net equity | New best val net equity |
| --- | ---: | ---: |
| `prediction_move_top_train_ic_selected` | `1.022` | `0.925` |
| `prediction_move_top_20` | `1.333` | `0.693` |

Read: `market_leader_return K=3,w=60` gives a tiny prediction gain in the filter, but it hurts trade/cost behavior. It is not a model upgrade.

## Decision

- Do not use `K=25` as a core signal.
- Do not add market-leader return directly to the base LSTM.
- Do not promote leader hand-gates.
- Keep `market_leader_return` available as a documented research feature and compatibility replacement for `vingroup_momentum`.
- For improving the model, continue with the filter-signal and holding-period selector path; leader signals are secondary diagnostics unless a new train-only rule shows stable cross-artifact improvement.
