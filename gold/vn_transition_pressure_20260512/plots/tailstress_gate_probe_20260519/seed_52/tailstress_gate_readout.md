# Tailstress Gate Probe

Rules are fitted on train and evaluated on validation. Holdout/test is not used.

## Validation

| candidate | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `conservative_mean_positive_gap` | 0.03590 | 4.767% | 6.259% | 9.957% | 21 | 11 | 0.185 | 0.509 |
| `simplex_tail` | 0.03569 | 4.782% | 6.300% | 11.770% | 30 | 13 | 0.172 | 0.509 |
| `tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 | 0.505 |
| `simplex_rel` | 0.03198 | 4.807% | 6.289% | 11.152% | 33 | 10 | 0.158 | 0.510 |
| `gate_tailstress_abs` | 0.03175 | 4.766% | 6.282% | 12.653% | 34 | 16 | 0.208 | 0.502 |
| `gate_disagreement` | 0.03171 | 4.770% | 6.281% | 12.653% | 34 | 16 | 0.200 | 0.500 |
| `conservative_tailstress_abs` | 0.03132 | 4.804% | 6.278% | 10.516% | 24 | 9 | 0.175 | 0.508 |
| `conservative_disagreement` | 0.02936 | 4.801% | 6.280% | 10.516% | 24 | 9 | 0.187 | 0.510 |
| `gate_mean_positive_gap` | 0.02751 | 4.804% | 6.284% | 12.653% | 36 | 14 | 0.202 | 0.507 |
| `tail_loss` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 | 0.504 |
| `weighted` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 | 0.504 |
| `tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 | 0.141 | 0.498 |
| `base` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 | 0.496 |
| `instance` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 | 0.493 |

## Fitted Parameters

```json
{
  "simplex_rel": {
    "tail_loss": 0.30000000000000004,
    "tailstress": 0.5,
    "tailstress_past": 0.0,
    "base": 0.2,
    "instance": 0.0
  },
  "simplex_tail": {
    "tail_loss": 0.19999999999999996,
    "tailstress": 0.7,
    "tailstress_past": 0.0,
    "base": 0.09999999999999998,
    "instance": 0.0
  },
  "gate_disagreement": [
    0.0073830868222499995,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_disagreement": [
    0.006603975072000002,
    "tail_loss",
    "tailstress"
  ],
  "gate_tailstress_abs": [
    0.007923300225,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_tailstress_abs": [
    0.00869574964,
    "tail_loss",
    "tailstress"
  ],
  "gate_mean_positive_gap": [
    -0.0002689289989477551,
    "tailstress",
    "tail_loss"
  ],
  "conservative_gate_mean_positive_gap": [
    0.003031465018470665,
    "tail_loss",
    "tailstress"
  ]
}
```

## Read

- A gate is useful only if it improves `rel_score` without increasing daily max/spike count.
- Direct `tailstress` remains a diagnostic signal if it wins `rel_score` but worsens spikes.
- If a train-fitted gate cannot beat `tail_loss`, tailstress should not be injected directly into the production LSTM yet.