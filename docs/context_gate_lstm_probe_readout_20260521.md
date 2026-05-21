# Context Gate LSTM Probe Readout 2026-05-21

Scope: seed-52 validation screen, current VN LSTM setup.

Goal: keep LSTM as the main forecaster, but let the model learn when to trust market-context features.

## Tested variants

### 1. Hard context gate

```text
stock_features   -> stock LSTM -> stock_hidden
context_features -> context LSTM -> context_hidden
gate = sigmoid(W context_hidden)
final_hidden = concat(stock_hidden, gate * context_hidden_projected)
```

Context features:

```text
vnindex_return, market_leader_return, a_d_ratio,
sector_momentum_rank, is_top_2_sector
```

### 2. Residual context gate

```text
all_features -> full LSTM -> hidden
context_features -> context LSTM -> context_delta
gate = sigmoid(W context_hidden)
final_hidden = hidden + gate * context_delta
```

This version is softer because the full LSTM path remains intact.

## Results

Validation only, seed 52.

| case | rel_score | q90 abs E | daily q90 p90 | daily q90 max | DA | pred/actual q90 | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline stressaux_w20 | 0.03300 | 4.80% | 6.37% | 9.62% | 51.05% | 0.171 | 29 | 12 |
| hard context gate | 0.00297 | 4.98% | 6.56% | 8.21% | 48.37% | 0.098 | 43 | 1 |
| residual context gate | 0.02093 | 4.83% | 6.47% | 10.93% | 49.84% | 0.185 | 41 | 14 |
| context dropout 20% | 0.02125 | 4.86% | 6.42% | 9.88% | 49.98% | 0.136 | 32 | 10 |

## Decision

Do not promote context gate yet.

Reason:

- Hard gate lowers extreme spike count but collapses prediction amplitude and directional accuracy.
- Residual gate avoids total collapse, but still loses rel_score and worsens high-error days.
- The baseline stressaux LSTM remains the better all-days forecaster.

## Interpretation

The idea is still valid, but this implementation is too weakly constrained:

```text
The gate learned risk shrinkage, not better context timing.
```

For a stronger next attempt, the gate should be supervised or constrained by a target that directly measures when context is useful, for example:

```text
context_useful_label = 1 if market-context model improves error over stock-only model on train/calibration
```

That would make the gate learn "when context helps", not just "when to shrink".

Artifacts:

- `gold/vn_transition_pressure_20260512/plots/context_gate_lstm_probe_20260521/seed_52_screen/summary.md`
- `gold/vn_transition_pressure_20260512/plots/context_gate_lstm_probe_20260521/seed_52_residual_screen/summary.md`
- `data/processed/assets/data_info_vn/history/training_runs/reports/context_gate_lstm_probe_20260521/seed52_context_gate_comparison_v2.csv`
