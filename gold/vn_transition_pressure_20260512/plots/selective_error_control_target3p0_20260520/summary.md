# Selective Error-Control Readout

Base prediction: `stressaux_w20`. Risk scores are fitted on early train, calibrated on late train, evaluated on validation.
Target: validation-style accepted daily `q90(|E|)` p90 near `3.0%`.

## Best Frontier Rows

| score | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `risk_logistic` | `coverage_q30` | 10.84% | 83.81% | -0.00821 | 2.25% | 1.91% | 3.09% | 5.70% | 8.3 | 1.3 | 0.0 |
| `risk_hgb` | `coverage_q30` | 11.57% | 97.57% | -0.00826 | 2.28% | 1.91% | 3.23% | 5.80% | 7.0 | 2.0 | 0.0 |
| `risk_logistic` | `coverage_q40` | 16.54% | 88.87% | 0.00496 | 2.45% | 2.04% | 3.35% | 5.78% | 17.7 | 3.3 | 0.0 |
| `risk_hgb` | `coverage_q40` | 17.05% | 98.43% | 0.00526 | 2.47% | 2.05% | 3.37% | 5.80% | 17.0 | 3.3 | 0.0 |
| `risk_input_noise` | `coverage_q30` | 13.29% | 97.82% | 0.00429 | 2.46% | 2.01% | 3.40% | 5.53% | 10.3 | 1.3 | 0.0 |
| `risk_input_noise` | `coverage_q40` | 19.77% | 99.54% | 0.00479 | 2.61% | 2.19% | 3.56% | 6.58% | 28.0 | 4.7 | 0.0 |
| `risk_logistic` | `calibrated_p90_3p5` | 21.47% | 90.84% | 0.00453 | 2.55% | 2.12% | 3.56% | 5.92% | 32.7 | 4.7 | 0.0 |
| `risk_hgb` | `calibrated_p90_3p5` | 21.86% | 99.04% | 0.00366 | 2.58% | 2.14% | 3.58% | 6.13% | 30.3 | 5.7 | 0.0 |
| `risk_input_noise` | `calibrated_p90_3p5` | 20.98% | 99.65% | 0.00696 | 2.63% | 2.21% | 3.59% | 6.33% | 32.7 | 5.3 | 0.0 |
| `risk_logistic` | `coverage_q50` | 24.15% | 91.96% | 0.00268 | 2.61% | 2.19% | 3.64% | 5.98% | 40.7 | 5.0 | 0.0 |
| `risk_hgb` | `coverage_q50` | 24.14% | 99.24% | 0.00437 | 2.63% | 2.22% | 3.70% | 6.25% | 37.3 | 6.3 | 0.0 |
| `risk_hybrid` | `coverage_q30` | 29.30% | 97.27% | 0.00643 | 2.75% | 2.35% | 3.72% | 6.80% | 52.0 | 8.3 | 0.0 |
| `risk_input_noise` | `coverage_q50` | 27.54% | 100.00% | 0.00408 | 2.78% | 2.31% | 3.86% | 6.50% | 61.3 | 9.3 | 0.0 |
| `risk_logistic` | `coverage_q60` | 33.79% | 94.54% | 0.00155 | 2.81% | 2.41% | 3.90% | 6.79% | 70.3 | 12.0 | 0.0 |
| `risk_hgb` | `coverage_q60` | 33.27% | 99.85% | 0.00247 | 2.84% | 2.41% | 3.91% | 7.07% | 60.7 | 14.0 | 0.0 |
| `risk_logistic` | `coverage_q70` | 45.43% | 96.66% | 0.00247 | 3.01% | 2.64% | 4.07% | 7.10% | 100.0 | 22.3 | 0.0 |
| `risk_hybrid` | `coverage_q40` | 38.60% | 99.14% | 0.00478 | 2.92% | 2.55% | 4.09% | 6.98% | 82.7 | 18.7 | 0.0 |
| `risk_input_noise` | `coverage_q60` | 37.25% | 100.00% | 0.00410 | 2.96% | 2.53% | 4.11% | 7.00% | 95.0 | 21.0 | 0.0 |
| `risk_hybrid` | `calibrated_p90_3p5` | 45.30% | 99.70% | 0.00453 | 3.04% | 2.68% | 4.19% | 7.02% | 115.3 | 24.0 | 0.0 |
| `risk_input_noise` | `coverage_q70` | 48.56% | 100.00% | 0.00364 | 3.17% | 2.76% | 4.22% | 7.17% | 131.3 | 31.3 | 0.0 |
| `risk_hybrid` | `coverage_q50` | 48.66% | 99.85% | 0.00522 | 3.10% | 2.73% | 4.22% | 7.09% | 126.7 | 27.0 | 0.0 |
| `risk_hgb` | `coverage_q70` | 45.84% | 99.95% | 0.00409 | 3.10% | 2.66% | 4.30% | 7.29% | 115.7 | 27.7 | 0.0 |
| `risk_disagreement` | `lowest_risk_available` | 20.00% | 92.49% | 0.00087 | 3.20% | 2.65% | 4.40% | 6.98% | 63.5 | 18.5 | 0.0 |
| `risk_disagreement` | `calibrated_p90_3p5` | 25.00% | 94.08% | 0.00705 | 3.14% | 2.63% | 4.41% | 7.01% | 73.0 | 20.0 | 0.0 |

## Calibration Thresholds

|   seed | score             |   threshold | policy                |
|-------:|:------------------|------------:|:----------------------|
|     43 | risk_hgb          |   0.0805273 | calibrated_p90_3p5    |
|     43 | risk_logistic     |   0.365637  | calibrated_p90_3p5    |
|     43 | risk_hybrid       |   0.395705  | calibrated_p90_3p5    |
|     43 | risk_input_noise  |   0.394194  | calibrated_p90_3p5    |
|     43 | risk_disagreement |   0.250017  | calibrated_p90_3p5    |
|     52 | risk_hgb          |   0.100391  | calibrated_p90_3p5    |
|     52 | risk_logistic     |   0.394846  | calibrated_p90_3p5    |
|     52 | risk_hybrid       |   0.484125  | calibrated_p90_3p5    |
|     52 | risk_input_noise  |   0.419074  | calibrated_p90_3p5    |
|     52 | risk_disagreement |   0.200018  | lowest_risk_available |
|     71 | risk_hgb          |   0.0988552 | calibrated_p90_3p5    |
|     71 | risk_logistic     |   0.369698  | calibrated_p90_3p5    |
|     71 | risk_hybrid       |   0.486842  | calibrated_p90_3p5    |
|     71 | risk_input_noise  |   0.399157  | calibrated_p90_3p5    |
|     71 | risk_disagreement |   0.200018  | lowest_risk_available |

## Read

- Full coverage stressaux_w20: rel_score `0.02477`, daily p90 `6.39%`, days >=3.5% `363.3`.
- Best selective row: `risk_logistic/calibrated_p90_3p5` with obs coverage `21.5%`, rel_score `0.00453`, daily p90 `3.56%`.
- If the 3.5% target is only reached at low coverage, the right paper framing is selective prediction/error-control, not full-coverage forecasting.
- If input-noise or disagreement scores rank well, the next model improvement should move those features into a learned confidence head and cleaner input normalization.