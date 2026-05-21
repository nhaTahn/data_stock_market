# Feature Redundancy + LSTM Probe Readout 2026-05-20

Scope: current VN LSTM feature set, train-only redundancy diagnostics plus seed-52 validation probe.

## 1. Train-only feature redundancy

Current default LSTM feature count: 26.

Only one strongly redundant pair appears under train-only Pearson correlation:

| feature_a | feature_b | corr |
|:--|:--|--:|
| `market_leader_return` | `vnindex_return` | 0.868 |

Important read:

- The current feature set is already mostly pruned.
- The problem is not many duplicate technical indicators anymore.
- The only obvious overlap is market-wide beta: top-liquid/leader return is highly correlated with equal-weight market proxy.

Top train-only feature-target daily IC signals:

| feature | daily IC mean | interpretation |
|:--|--:|:--|
| `close_position` | -0.0818 | strong candle-position mean reversion |
| `lower_shadow` | -0.0762 | lower wick often mean-reverts next day |
| `selling_pressure` | 0.0707 | Wyckoff pressure has useful structure |
| `volume_ratio_20` | 0.0486 | abnormal volume has weak signal |
| `upper_shadow` | 0.0485 | upper wick contains reversal/pressure signal |

Artifact:

- `gold/vn_transition_pressure_20260512/plots/current_lstm_feature_redundancy_20260520/summary.md`

## 2. LSTM probe

All rows use seed 52, same stressaux LSTM variant, validation only.

| case | rel_score | q90 abs E | daily q90 p90 | daily q90 max | DA | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.03300 | 4.80% | 6.37% | 9.62% | 51.05% | 29 | 12 |
| drop `market_leader_return` | 0.02029 | 4.89% | 6.58% | 11.16% | 49.65% | 38 | 12 |
| replace with `market_leader_excess_return` | 0.02115 | 4.91% | 6.58% | 11.58% | 49.42% | 45 | 15 |

Decision:

- Do not drop `market_leader_return`.
- Do not replace it with `market_leader_excess_return` for the current LSTM setup.
- The raw leader-return feature is redundant by correlation, but still useful to the LSTM. This means correlation alone is not sufficient to prune market-context features.

## 3. What this answers

Question: "Feature selection có noise / trùng nhau không?"

Answer:

```text
Có một overlap lớn: market_leader_return ~ vnindex_return.
Nhưng khi train thử, bỏ hoặc orthogonalize feature này làm mô hình xấu hơn.
Vì vậy hiện tại không nên cải thiện bằng pruning thô.
```

Question: "Nếu không phải feature trùng thì phát triển tiếp sao?"

Next useful direction:

```text
Giữ feature set hiện tại.
Tập trung vào feature interaction / regime conditioning thay vì drop feature.
Ưu tiên học khi nào market_leader_return hữu ích, không phải loại nó.
```

Possible next experiments:

1. Conditional market context gate:
   `leader_context_weight = g(vnindex_return, market_leader_return, a_d_ratio, volatility_20)`.
2. Feature dropout / input denoising during LSTM training:
   randomly mask correlated market-context features during training to reduce dependence on one proxy.
3. Multi-seed confirmation only if a new method beats baseline on seed 52 first.
