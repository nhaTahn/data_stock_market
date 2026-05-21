# Mathematical Report: VN Transition Pressure Policy

## 1. Data

Let:

```text
X_raw = raw VN market data
X_process = F(X_raw)
```

For stock `i` at trading day `t`:

```text
X_{i,t} in R^{15 x 29}
y_{i,t+1} = r_{i,t+1}
```

where `r_{i,t+1}` is the next-day return.

## 2. Base Forecast Model

The frozen base model is:

```text
broad_signmag_portable_no_identity_20260428_allvn_r01 / lstm_seed_52
```

Base LSTM:

```text
h_{i,t} = LSTM_{64,32}(X_{i,t})
f_{i,t+1} = W h_{i,t} + b
```

Here `f_{i,t+1}` is the base forecast for next-day return.

Main score:

```text
rel_score = 1 - loss(y - f) / loss(y)
loss(z) = Q_0.5(|z|) + 0.5 Q_0.9(|z|)
```

## 3. Filter Signal

A small LSTM filter receives transformed prediction/context features:

```text
Z_{i,t} = G(f_{i,t}, market_context_t, technical_context_{i,t})
p_{i,t} = FilterLSTM(Z_{i,t-9:t})
```

where `p_{i,t}` estimates the probability that the base forecast is tradeable.

The fresh stress artifact excludes:

```text
market_leader_return
```

so the frozen result is not dependent on the market-leader feature.

## 4. Committee Selector

For each candidate policy `c`, the validation engine selects a holding-period rule using train-only rolling folds.

The candidate score is based on forecast strength and filter confidence:

```text
s_{i,t}^{move} = |f_{i,t+1}| p_{i,t}
```

The selected daily candidate set is:

```text
C_t = SelectTop(s_{i,t}, constraints)
```

with transaction cost and turnover evaluated explicitly.

## 5. Wyckoff Phase Gate

Let:

```text
phi_t in {accumulation, markup, distribution, markdown, transition}
```

be the Wyckoff-style market phase.

The frozen gate is:

```text
g(phi_t) =
  legacy_filter_shortlist,  if phi_t = accumulation
  all_committee_candidates, if phi_t in {markup, distribution, markdown}
  transition_rule,          if phi_t = transition
```

## 6. Transition Pressure Risk Filter

Define:

```text
pressure_delta_20(t) = MA20(mean_i(buying_pressure_{i,t} - selling_pressure_{i,t}))
```

During transition phase:

```text
transition_rule =
  all_committee_candidates, if pressure_delta_20(t) >= 0
  cash,                     otherwise
```

The frozen policy name is:

```text
vn_legacy_acc_all_else_transition_pressure_nonneg
```

## 7. Portfolio Execution

Advisor-facing frozen variant:

```text
min_positions = 6
cost = 15 bps per unit turnover
```

Let `w_{i,t}` be portfolio weight. Turnover is:

```text
TO_t = sum_i |w_{i,t} - w_{i,t-1}|
```

Net daily return:

```text
R_t^{net} = sum_i w_{i,t} r_{i,t+1} - 0.0015 TO_t
```

## 8. Validation Result

Validation protocol:

```text
strict non-overlap rolling validation
train = 126 days
test = 21 days
step = 21 days
holdout/test = not used
```

Four-artifact result for `min_positions=6`:

| Metric | Value |
| --- | ---: |
| positive artifacts | 4 / 4 |
| worst-artifact equity | 1.186 |
| mean equity | 1.460 |
| minimum Sharpe | +0.55 |
| worst max drawdown | -21.7% |
| maximum turnover | 0.179 |

## 9. Interpretation

The frozen architecture remains LSTM-centered.

There are two learned LSTM components:

```text
Base LSTM:        X_{i,t} -> f_{i,t+1}
Filter LSTM:      Z_{i,t-9:t} -> p_{i,t}
```

The committee selector and Wyckoff/pressure gates are not replacement models. They are execution and risk-control layers built around the LSTM signal:

```text
LSTM forecast -> LSTM filter -> committee -> regime gate -> risk-controlled execution
```

The current result should therefore be described as an LSTM-based forecasting and filtering system with a rule-based execution layer.

The result is weaker after the fresh no-leader stress artifact, but it remains positive across all four validation artifacts and passes the risk-control screen.

This is suitable for advisor discussion as a VN-first research candidate, not as a production-ready model.

The next valid step is leakage audit, then one controlled holdout read without further tuning.
