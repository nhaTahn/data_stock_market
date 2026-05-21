# Feature Denoise LSTM Probe Readout 2026-05-20

Scope: seed-52 validation screen after feature redundancy analysis.

Hypothesis:

```text
If market-context features are correlated/noisy, random masking during training
may force LSTM to learn more robust combinations instead of depending on one proxy.
```

Implementation:

- keep full validation input unchanged;
- during training only, randomly set selected context feature values to 0 after scaling;
- context columns: `vnindex_return`, `market_leader_return`, `a_d_ratio`, `sector_momentum_rank`, `is_top_2_sector`.

## Results

| case | rel_score | q90 abs E | daily q90 p90 | daily q90 max | DA | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0.03300 | 4.80% | 6.37% | 9.62% | 51.05% | 29 | 12 |
| drop `market_leader_return` | 0.02029 | 4.89% | 6.58% | 11.16% | 49.65% | 38 | 12 |
| replace with `market_leader_excess_return` | 0.02115 | 4.91% | 6.58% | 11.58% | 49.42% | 45 | 15 |
| context dropout 10% | 0.02682 | 4.81% | 6.42% | 11.34% | 50.75% | 41 | 11 |
| context dropout 20% | 0.02125 | 4.86% | 6.42% | 9.88% | 49.98% | 32 | 10 |

## Decision

Do not promote feature pruning, leader-excess replacement, or context dropout.

Reason:

- All variants reduce validation `rel_score` versus baseline.
- `ctxdrop20` lowers days >=8% from 12 to 10, but this comes from lower prediction amplitude and weaker directional accuracy.
- The best current LSTM still needs the raw market-context features.

## Research conclusion

Feature redundancy is not the main bottleneck right now.

The current feature set is already fairly clean. The remaining problem is that market-context signals are useful but unstable across regimes. The next improvement should be conditional usage of context rather than deletion:

```text
return_hat = LSTM(stock_features, context_features)
context_gate = g(context_state)
final_hidden = combine(stock_hidden, context_gate * context_hidden)
```

This keeps LSTM as the core model but lets it learn when to trust market-context information.

Artifacts:

- `gold/vn_transition_pressure_20260512/plots/current_lstm_feature_redundancy_20260520/summary.md`
- `gold/vn_transition_pressure_20260512/plots/feature_prune_lstm_probe_20260520/seed52_feature_prune_comparison.csv`
- `gold/vn_transition_pressure_20260512/plots/feature_denoise_lstm_probe_20260520/ctxdrop_seed52/summary.md`
