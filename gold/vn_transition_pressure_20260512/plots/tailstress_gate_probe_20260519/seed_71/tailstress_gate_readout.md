# Tailstress Gate Probe

Rules are fitted on train and evaluated on validation. Holdout/test is not used.

## Validation

| candidate | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `conservative_mean_positive_gap` | 0.01909 | 4.883% | 6.320% | 8.466% | 19 | 3 | 0.134 | 0.491 |
| `tailstress` | 0.01896 | 4.885% | 6.320% | 8.466% | 19 | 3 | 0.134 | 0.491 |
| `gate_tailstress_abs` | 0.01812 | 4.876% | 6.282% | 8.672% | 21 | 4 | 0.155 | 0.496 |
| `gate_disagreement` | 0.01752 | 4.879% | 6.293% | 8.466% | 20 | 3 | 0.148 | 0.495 |
| `base` | 0.01744 | 4.927% | 6.606% | 9.534% | 44 | 16 | 0.123 | 0.494 |
| `conservative_disagreement` | 0.01584 | 4.911% | 6.461% | 9.084% | 32 | 4 | 0.148 | 0.497 |
| `gate_mean_positive_gap` | 0.01525 | 4.910% | 6.488% | 9.084% | 32 | 4 | 0.146 | 0.498 |
| `tail_loss` | 0.01482 | 4.912% | 6.477% | 9.084% | 32 | 4 | 0.150 | 0.496 |
| `simplex_tail` | 0.01417 | 4.947% | 6.555% | 8.584% | 36 | 3 | 0.100 | 0.503 |
| `conservative_tailstress_abs` | 0.01372 | 4.928% | 6.477% | 9.084% | 24 | 3 | 0.127 | 0.491 |
| `simplex_rel` | 0.01219 | 4.967% | 6.427% | 8.397% | 25 | 1 | 0.109 | 0.501 |
| `instance` | 0.00889 | 4.990% | 6.554% | 8.615% | 35 | 3 | 0.133 | 0.492 |
| `tailstress_past` | 0.00748 | 4.977% | 6.646% | 7.953% | 37 | 0 | 0.085 | 0.492 |
| `weighted` | 0.00391 | 4.974% | 6.523% | 7.992% | 45 | 0 | 0.091 | 0.484 |

## Fitted Parameters

```json
{
  "simplex_rel": {
    "tail_loss": 0.1,
    "tailstress": 0.4,
    "tailstress_past": 0.0,
    "base": 0.0,
    "instance": 0.5
  },
  "simplex_tail": {
    "tail_loss": 0.3,
    "tailstress": 0.09999999999999998,
    "tailstress_past": 0.19999999999999996,
    "base": 0.3,
    "instance": 0.09999999999999998
  },
  "gate_disagreement": [
    0.00882207462975,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_disagreement": [
    0.0032426180400000003,
    "tail_loss",
    "tailstress"
  ],
  "gate_tailstress_abs": [
    0.008127136629999998,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_tailstress_abs": [
    0.010284140890000001,
    "tail_loss",
    "tailstress"
  ],
  "gate_mean_positive_gap": [
    0.000872045069855899,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_mean_positive_gap": [
    0.0020604809751298293,
    "tail_loss",
    "tailstress"
  ]
}
```

## Read

- A gate is useful only if it improves `rel_score` without increasing daily max/spike count.
- Direct `tailstress` remains a diagnostic signal if it wins `rel_score` but worsens spikes.
- If a train-fitted gate cannot beat `tail_loss`, tailstress should not be injected directly into the production LSTM yet.