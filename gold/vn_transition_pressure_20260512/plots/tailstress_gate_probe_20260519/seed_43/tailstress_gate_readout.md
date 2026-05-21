# Tailstress Gate Probe

Rules are fitted on train and evaluated on validation. Holdout/test is not used.

## Validation

| candidate | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `weighted` | 0.02968 | 4.831% | 6.460% | 10.508% | 40 | 10 | 0.150 | 0.509 |
| `gate_mean_positive_gap` | 0.02937 | 4.806% | 6.358% | 10.549% | 40 | 13 | 0.183 | 0.511 |
| `gate_tailstress_abs` | 0.02831 | 4.817% | 6.370% | 10.549% | 32 | 15 | 0.189 | 0.508 |
| `simplex_rel` | 0.02818 | 4.826% | 6.378% | 10.131% | 35 | 10 | 0.158 | 0.511 |
| `tailstress` | 0.02652 | 4.830% | 6.265% | 10.549% | 34 | 14 | 0.179 | 0.510 |
| `gate_disagreement` | 0.02582 | 4.827% | 6.345% | 10.549% | 33 | 15 | 0.181 | 0.508 |
| `tail_loss` | 0.02580 | 4.836% | 6.405% | 10.007% | 37 | 11 | 0.168 | 0.508 |
| `simplex_tail` | 0.02580 | 4.836% | 6.405% | 10.007% | 37 | 11 | 0.168 | 0.508 |
| `conservative_disagreement` | 0.02569 | 4.839% | 6.439% | 10.007% | 37 | 11 | 0.167 | 0.508 |
| `conservative_tailstress_abs` | 0.02527 | 4.847% | 6.384% | 10.007% | 39 | 11 | 0.158 | 0.506 |
| `conservative_mean_positive_gap` | 0.02339 | 4.860% | 6.378% | 10.007% | 32 | 10 | 0.163 | 0.506 |
| `base` | 0.01719 | 4.918% | 6.497% | 9.361% | 39 | 14 | 0.142 | 0.505 |
| `tailstress_past` | 0.00692 | 4.950% | 6.480% | 8.361% | 35 | 6 | 0.105 | 0.491 |
| `instance` | 0.00448 | 5.010% | 6.571% | 8.183% | 36 | 2 | 0.116 | 0.496 |

## Fitted Parameters

```json
{
  "simplex_rel": {
    "tail_loss": 0.8,
    "tailstress": 0.2,
    "tailstress_past": 0.0,
    "base": 0.0,
    "instance": 0.0
  },
  "simplex_tail": {
    "tail_loss": 1.0,
    "tailstress": 0.0,
    "tailstress_past": 0.0,
    "base": 0.0,
    "instance": 0.0
  },
  "gate_disagreement": [
    0.009808732006500002,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_disagreement": [
    0.0034722364900000025,
    "tail_loss",
    "tailstress"
  ],
  "gate_tailstress_abs": [
    0.013660559639999998,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_tailstress_abs": [
    0.0049698457750000005,
    "tail_loss",
    "tailstress"
  ],
  "gate_mean_positive_gap": [
    0.0014616383970261317,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_mean_positive_gap": [
    0.00031526980005499986,
    "tail_loss",
    "tailstress"
  ]
}
```

## Read

- A gate is useful only if it improves `rel_score` without increasing daily max/spike count.
- Direct `tailstress` remains a diagnostic signal if it wins `rel_score` but worsens spikes.
- If a train-fitted gate cannot beat `tail_loss`, tailstress should not be injected directly into the production LSTM yet.