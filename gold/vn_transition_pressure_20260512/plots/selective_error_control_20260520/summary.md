# Selective Error-Control Readout

Base prediction: `stressaux_w20`. Risk scores are fitted on early train, calibrated on late train, evaluated on validation.
Target: validation-style accepted daily `q90(|E|)` p90 near `3.5%`.

## Best Frontier Rows

| score | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `risk_logistic` | `coverage_q30` | 10.63% | 82.75% | -0.00938 | 2.25% | 1.90% | 3.19% | 5.73% | 9.3 | 1.7 | 0.0 |
| `risk_hgb` | `coverage_q30` | 11.25% | 95.75% | -0.00335 | 2.29% | 1.91% | 3.25% | 5.63% | 8.7 | 2.3 | 0.0 |
| `risk_logistic` | `coverage_q40` | 16.34% | 88.01% | 0.00471 | 2.44% | 2.03% | 3.37% | 5.77% | 17.0 | 3.7 | 0.0 |
| `risk_input_noise` | `coverage_q30` | 13.29% | 97.82% | 0.00429 | 2.46% | 2.01% | 3.40% | 5.53% | 10.3 | 1.3 | 0.0 |
| `risk_hgb` | `coverage_q40` | 16.74% | 98.28% | 0.00528 | 2.47% | 2.10% | 3.43% | 6.14% | 18.0 | 5.0 | 0.0 |
| `risk_input_noise` | `coverage_q40` | 19.77% | 99.54% | 0.00479 | 2.61% | 2.19% | 3.56% | 6.58% | 28.0 | 4.7 | 0.0 |
| `risk_logistic` | `coverage_q50` | 23.80% | 91.05% | 0.00353 | 2.60% | 2.17% | 3.59% | 6.05% | 38.0 | 4.7 | 0.0 |
| `risk_hgb` | `coverage_q50` | 23.58% | 99.60% | 0.00470 | 2.63% | 2.22% | 3.69% | 6.20% | 35.7 | 7.3 | 0.0 |
| `risk_hybrid` | `coverage_q30` | 29.55% | 96.86% | 0.00708 | 2.75% | 2.39% | 3.77% | 6.84% | 54.7 | 10.7 | 0.0 |
| `risk_input_noise` | `coverage_q50` | 27.54% | 100.00% | 0.00408 | 2.78% | 2.31% | 3.86% | 6.50% | 61.3 | 9.3 | 0.0 |
| `risk_logistic` | `coverage_q60` | 33.43% | 94.03% | 0.00236 | 2.80% | 2.39% | 3.89% | 6.80% | 67.3 | 11.7 | 0.0 |
| `risk_hgb` | `coverage_q60` | 32.54% | 99.95% | 0.00462 | 2.83% | 2.40% | 3.89% | 6.98% | 55.7 | 10.3 | 0.0 |
| `risk_logistic` | `coverage_q70` | 45.10% | 96.00% | 0.00285 | 3.00% | 2.61% | 4.04% | 7.11% | 102.7 | 20.3 | 0.0 |
| `risk_logistic` | `calibrated_p90_3p5` | 47.30% | 96.46% | 0.00412 | 3.04% | 2.66% | 4.06% | 7.17% | 109.7 | 24.7 | 0.0 |
| `risk_hybrid` | `coverage_q40` | 38.76% | 98.99% | 0.00397 | 2.93% | 2.54% | 4.07% | 7.02% | 85.7 | 18.3 | 0.0 |
| `risk_input_noise` | `coverage_q60` | 37.25% | 100.00% | 0.00410 | 2.96% | 2.53% | 4.11% | 7.00% | 95.0 | 21.0 | 0.0 |
| `risk_input_noise` | `calibrated_p90_3p5` | 39.03% | 100.00% | 0.00441 | 2.99% | 2.58% | 4.13% | 7.00% | 102.3 | 23.3 | 0.0 |
| `risk_hybrid` | `coverage_q50` | 48.49% | 99.95% | 0.00583 | 3.11% | 2.72% | 4.19% | 7.09% | 126.0 | 24.3 | 0.0 |
| `risk_input_noise` | `coverage_q70` | 48.56% | 100.00% | 0.00364 | 3.17% | 2.76% | 4.22% | 7.17% | 131.3 | 31.3 | 0.0 |
| `risk_hgb` | `calibrated_p90_3p5` | 45.47% | 100.00% | 0.00558 | 3.08% | 2.65% | 4.23% | 7.27% | 118.7 | 29.7 | 0.0 |
| `risk_hgb` | `coverage_q70` | 45.47% | 100.00% | 0.00558 | 3.08% | 2.65% | 4.23% | 7.27% | 118.7 | 29.7 | 0.0 |
| `risk_logistic` | `coverage_q80` | 58.91% | 98.18% | 0.00452 | 3.29% | 2.89% | 4.42% | 7.49% | 151.3 | 36.7 | 0.0 |
| `risk_disagreement` | `coverage_q30` | 30.00% | 95.90% | 0.00415 | 3.27% | 2.79% | 4.52% | 7.06% | 99.0 | 26.0 | 0.0 |
| `risk_hybrid` | `coverage_q60` | 58.97% | 100.00% | 0.00295 | 3.33% | 2.91% | 4.59% | 7.29% | 166.7 | 40.7 | 0.0 |

## Calibration Thresholds

|   seed | score             |   threshold | policy             |
|-------:|:------------------|------------:|:-------------------|
|     43 | risk_hgb          |    0.109126 | calibrated_p90_3p5 |
|     43 | risk_logistic     |    0.502827 | calibrated_p90_3p5 |
|     43 | risk_hybrid       |    0.681203 | calibrated_p90_3p5 |
|     43 | risk_input_noise  |    0.495439 | calibrated_p90_3p5 |
|     43 | risk_disagreement |    0.500012 | calibrated_p90_3p5 |
|     52 | risk_hgb          |    0.110322 | calibrated_p90_3p5 |
|     52 | risk_logistic     |    0.503108 | calibrated_p90_3p5 |
|     52 | risk_hybrid       |    0.728006 | calibrated_p90_3p5 |
|     52 | risk_input_noise  |    0.523703 | calibrated_p90_3p5 |
|     52 | risk_disagreement |    0.55001  | calibrated_p90_3p5 |
|     71 | risk_hgb          |    0.113454 | calibrated_p90_3p5 |
|     71 | risk_logistic     |    0.54893  | calibrated_p90_3p5 |
|     71 | risk_hybrid       |    0.728294 | calibrated_p90_3p5 |
|     71 | risk_input_noise  |    0.500738 | calibrated_p90_3p5 |
|     71 | risk_disagreement |    0.55001  | calibrated_p90_3p5 |

## Read

- Full coverage stressaux_w20: rel_score `0.02477`, daily p90 `6.39%`, days >=3.5% `363.3`.
- Best selective row: `risk_logistic/calibrated_p90_3p5` with obs coverage `47.3%`, rel_score `0.00412`, daily p90 `4.06%`.
- If the 3.5% target is only reached at low coverage, the right paper framing is selective prediction/error-control, not full-coverage forecasting.
- If input-noise or disagreement scores rank well, the next model improvement should move those features into a learned confidence head and cleaner input normalization.