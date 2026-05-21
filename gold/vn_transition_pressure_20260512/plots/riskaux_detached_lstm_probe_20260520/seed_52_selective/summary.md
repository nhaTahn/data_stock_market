# RiskAux Selective Readout

Risk score comes from an LSTM auxiliary head. Detached variants stop risk-head gradients from updating the return backbone.

| variant | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q20` | 17.72% | 89.53% | 0.00068 | 2.82% | 2.34% | 3.81% | 7.24% | 35.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q20` | 17.72% | 89.53% | 0.00068 | 2.82% | 2.34% | 3.81% | 7.24% | 35.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q30` | 25.63% | 93.78% | 0.00402 | 2.98% | 2.50% | 3.99% | 7.17% | 77.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q30` | 25.63% | 93.78% | 0.00402 | 2.98% | 2.50% | 3.99% | 7.17% | 77.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `calibrated_train_p90` | 29.44% | 94.99% | 0.00642 | 3.06% | 2.52% | 4.09% | 7.22% | 91.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `calibrated_train_p90` | 29.43% | 94.99% | 0.00646 | 3.06% | 2.52% | 4.09% | 7.22% | 91.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q40` | 33.43% | 96.21% | -0.00127 | 3.16% | 2.63% | 4.29% | 7.25% | 106.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q40` | 33.43% | 96.21% | -0.00120 | 3.16% | 2.64% | 4.29% | 7.25% | 106.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q50` | 41.88% | 99.24% | -0.00337 | 3.30% | 2.78% | 4.47% | 7.19% | 138.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q50` | 41.88% | 99.24% | -0.00337 | 3.30% | 2.78% | 4.47% | 7.19% | 138.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q60` | 51.04% | 100.00% | -0.00135 | 3.46% | 2.90% | 4.86% | 7.27% | 170.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q60` | 51.04% | 100.00% | -0.00138 | 3.46% | 2.90% | 4.86% | 7.27% | 170.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q70` | 60.80% | 100.00% | 0.00161 | 3.66% | 3.03% | 5.14% | 7.26% | 202.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q70` | 60.80% | 100.00% | 0.00161 | 3.66% | 3.03% | 5.14% | 7.26% | 202.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q80` | 71.55% | 100.00% | 0.00307 | 3.86% | 3.19% | 5.55% | 7.46% | 247.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q80` | 71.55% | 100.00% | 0.00307 | 3.86% | 3.19% | 5.55% | 7.46% | 247.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q90` | 83.92% | 100.00% | -0.00040 | 4.15% | 3.40% | 6.05% | 7.66% | 300.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q90` | 83.92% | 100.00% | -0.00040 | 4.15% | 3.40% | 6.05% | 7.66% | 300.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `full_coverage` | 100.00% | 100.00% | 0.00350 | 4.99% | 3.68% | 6.49% | 8.62% | 362.0 | 9.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `full_coverage` | 100.00% | 100.00% | 0.00350 | 4.99% | 3.68% | 6.49% | 8.62% | 362.0 | 9.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w10` | `risk_q100` | 100.00% | 100.00% | 0.00354 | 4.99% | 3.68% | 6.49% | 8.62% | 362.0 | 9.0 |
| `plain_global_weighted_mild_tail35_riskaux_detached_w20` | `risk_q100` | 100.00% | 100.00% | 0.00354 | 4.99% | 3.68% | 6.49% | 8.62% | 362.0 | 9.0 |

## Read

- If RiskAux beats the post-hoc input-noise/disagreement scores at similar coverage, the learned confidence head is worth promoting.
- If it only matches post-hoc filters, the next improvement should focus on cleaner input features and stronger calibration rather than more heads.