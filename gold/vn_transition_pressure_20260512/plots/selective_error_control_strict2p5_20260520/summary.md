# Selective Error-Control Readout

Base prediction: `stressaux_w20`. Risk scores are fitted on early train, calibrated on late train, evaluated on validation.
Target: validation-style accepted daily `q90(|E|)` p90 near `2.5%`.

## Best Frontier Rows

| score | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `risk_hgb` | `coverage_q10` | 3.90% | 84.98% | -0.00835 | 1.75% | 1.73% | 1.77% | 1.79% | 0.0 | 0.0 | 0.0 |
| `risk_logistic` | `coverage_q10` | 2.63% | 59.59% | -0.01131 | 1.70% | 2.61% | 2.78% | 2.82% | 0.0 | 0.0 | 0.0 |
| `risk_logistic` | `lowest_risk_available` | 2.78% | 60.70% | -0.01836 | 1.70% | 2.32% | 2.82% | 2.95% | 0.0 | 0.0 | 0.0 |
| `risk_logistic` | `coverage_q30` | 10.75% | 84.22% | -0.00746 | 2.24% | 1.86% | 3.10% | 5.71% | 7.0 | 1.3 | 0.0 |
| `risk_hgb` | `coverage_q25` | 9.43% | 96.26% | -0.00207 | 2.17% | 1.88% | 3.17% | 6.01% | 5.7 | 3.3 | 0.0 |
| `risk_logistic` | `coverage_q25` | 8.31% | 81.44% | 0.00516 | 2.09% | 1.81% | 3.20% | 5.57% | 5.0 | 2.3 | 0.0 |
| `risk_hgb` | `calibrated_p90_3p5` | 5.11% | 89.73% | -0.00541 | 1.83% | 1.89% | 3.23% | 4.67% | 1.7 | 1.0 | 0.0 |
| `risk_hgb` | `coverage_q30` | 11.68% | 97.27% | -0.00891 | 2.28% | 1.93% | 3.29% | 5.84% | 8.7 | 2.7 | 0.0 |
| `risk_hybrid` | `calibrated_p90_3p5` | 14.11% | 89.18% | 0.00308 | 2.41% | 2.02% | 3.30% | 6.28% | 11.3 | 2.3 | 0.0 |
| `risk_hgb` | `coverage_q20` | 7.54% | 94.84% | 0.00518 | 2.03% | 1.80% | 3.31% | 6.22% | 4.7 | 2.3 | 0.0 |
| `risk_logistic` | `coverage_q40` | 16.45% | 88.92% | 0.00385 | 2.45% | 2.04% | 3.32% | 5.78% | 17.7 | 3.3 | 0.0 |
| `risk_input_noise` | `coverage_q25` | 10.40% | 95.80% | 0.00186 | 2.41% | 1.96% | 3.34% | 5.53% | 7.3 | 1.3 | 0.0 |
| `risk_hybrid` | `coverage_q10` | 10.98% | 85.79% | 0.00341 | 2.34% | 1.98% | 3.36% | 6.07% | 9.3 | 2.0 | 0.0 |
| `risk_hybrid` | `coverage_q15` | 15.62% | 91.25% | 0.00695 | 2.46% | 2.07% | 3.36% | 6.14% | 14.3 | 2.3 | 0.0 |
| `risk_input_noise` | `coverage_q30` | 13.29% | 97.82% | 0.00429 | 2.46% | 2.01% | 3.40% | 5.53% | 10.3 | 1.3 | 0.0 |
| `risk_hgb` | `coverage_q40` | 17.26% | 98.99% | 0.00582 | 2.47% | 2.10% | 3.44% | 5.81% | 17.0 | 3.7 | 0.0 |
| `risk_hgb` | `coverage_q15` | 5.70% | 91.60% | -0.00330 | 1.89% | 1.78% | 3.54% | 5.36% | 2.3 | 1.0 | 0.0 |
| `risk_input_noise` | `coverage_q40` | 19.77% | 99.54% | 0.00479 | 2.61% | 2.19% | 3.56% | 6.58% | 28.0 | 4.7 | 0.0 |
| `risk_hybrid` | `coverage_q20` | 20.20% | 93.83% | 0.00352 | 2.58% | 2.17% | 3.58% | 6.17% | 27.7 | 5.0 | 0.0 |
| `risk_input_noise` | `coverage_q20` | 7.75% | 93.12% | -0.00826 | 2.30% | 1.93% | 3.60% | 5.53% | 4.7 | 1.3 | 0.0 |
| `risk_logistic` | `coverage_q50` | 24.12% | 92.26% | 0.00381 | 2.60% | 2.20% | 3.61% | 5.88% | 40.0 | 4.7 | 0.0 |
| `risk_logistic` | `coverage_q20` | 6.20% | 76.63% | 0.00489 | 1.98% | 1.80% | 3.62% | 5.68% | 4.7 | 2.0 | 0.0 |
| `risk_hybrid` | `coverage_q25` | 24.65% | 95.75% | 0.00658 | 2.66% | 2.28% | 3.69% | 6.57% | 39.0 | 7.7 | 0.0 |
| `risk_hgb` | `coverage_q50` | 24.50% | 99.60% | 0.00535 | 2.65% | 2.22% | 3.72% | 6.43% | 37.0 | 7.0 | 0.0 |

## Calibration Thresholds

|   seed | score             |   threshold | policy                |
|-------:|:------------------|------------:|:----------------------|
|     43 | risk_hgb          |   0.073487  | calibrated_p90_3p5    |
|     43 | risk_logistic     |   0.266054  | calibrated_p90_3p5    |
|     43 | risk_hybrid       |   0.182159  | calibrated_p90_3p5    |
|     43 | risk_input_noise  |   0.239084  | calibrated_p90_3p5    |
|     43 | risk_disagreement |   0.100021  | lowest_risk_available |
|     52 | risk_hgb          |   0.0765805 | calibrated_p90_3p5    |
|     52 | risk_logistic     |   0.288873  | calibrated_p90_3p5    |
|     52 | risk_hybrid       |   0.188978  | calibrated_p90_3p5    |
|     52 | risk_input_noise  |   0.242464  | lowest_risk_available |
|     52 | risk_disagreement |   0.100021  | lowest_risk_available |
|     71 | risk_hgb          |   0.0630313 | calibrated_p90_3p5    |
|     71 | risk_logistic     |   0.248528  | lowest_risk_available |
|     71 | risk_hybrid       |   0.142739  | calibrated_p90_3p5    |
|     71 | risk_input_noise  |   0.242774  | calibrated_p90_3p5    |
|     71 | risk_disagreement |   0.100021  | lowest_risk_available |

## Read

- Full coverage stressaux_w20: rel_score `0.02477`, daily p90 `6.39%`, days >=3.5% `363.3`.
- Best selective row: `risk_logistic/lowest_risk_available` with obs coverage `2.8%`, rel_score `-0.01836`, daily p90 `2.82%`.
- If the 3.5% target is only reached at low coverage, the right paper framing is selective prediction/error-control, not full-coverage forecasting.
- If input-noise or disagreement scores rank well, the next model improvement should move those features into a learned confidence head and cleaner input normalization.