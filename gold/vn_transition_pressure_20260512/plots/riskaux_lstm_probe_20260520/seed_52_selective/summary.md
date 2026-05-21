# RiskAux Selective Readout

Risk score comes from the LSTM auxiliary head trained jointly with the return prediction head.

| variant | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=8% |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q20` | 8.60% | 88.47% | -0.00166 | 2.10% | 1.77% | 2.79% | 5.59% | 5.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q20` | 15.32% | 91.81% | -0.00207 | 2.68% | 2.02% | 3.44% | 6.93% | 19.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q30` | 15.09% | 93.17% | -0.00338 | 2.42% | 2.04% | 3.55% | 5.85% | 19.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q40` | 23.12% | 96.66% | -0.00209 | 2.64% | 2.24% | 3.66% | 7.12% | 42.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `calibrated_train_p90` | 32.33% | 98.63% | -0.00292 | 2.81% | 2.41% | 3.89% | 7.22% | 71.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q50` | 32.33% | 98.63% | -0.00292 | 2.81% | 2.41% | 3.89% | 7.22% | 71.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q60` | 42.47% | 99.24% | -0.00537 | 3.02% | 2.57% | 4.05% | 7.28% | 91.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q30` | 22.90% | 95.75% | -0.00406 | 2.89% | 2.29% | 4.09% | 7.22% | 64.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `calibrated_train_p90` | 30.71% | 98.79% | -0.00145 | 3.02% | 2.51% | 4.12% | 7.18% | 97.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q40` | 30.71% | 98.79% | -0.00145 | 3.02% | 2.51% | 4.12% | 7.18% | 97.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q50` | 39.45% | 100.00% | -0.00795 | 3.21% | 2.68% | 4.33% | 7.25% | 118.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q70` | 53.66% | 99.85% | -0.00924 | 3.24% | 2.80% | 4.33% | 7.34% | 132.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q60` | 49.07% | 100.00% | -0.00546 | 3.36% | 2.89% | 4.79% | 7.26% | 173.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q80` | 66.36% | 100.00% | -0.00811 | 3.50% | 3.03% | 4.93% | 7.41% | 193.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q70` | 59.13% | 100.00% | -0.00660 | 3.58% | 3.04% | 5.03% | 7.28% | 208.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q80` | 69.82% | 100.00% | -0.00170 | 3.80% | 3.18% | 5.45% | 7.29% | 241.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q90` | 80.56% | 100.00% | -0.00701 | 3.87% | 3.26% | 5.51% | 7.62% | 261.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q90` | 82.87% | 100.00% | -0.00841 | 4.15% | 3.37% | 6.08% | 7.60% | 292.0 | 0.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `full_coverage` | 100.00% | 100.00% | -0.00500 | 5.01% | 3.70% | 6.49% | 8.02% | 365.0 | 1.0 |
| `plain_global_weighted_mild_tail35_riskaux_w20` | `risk_q100` | 100.00% | 100.00% | -0.00498 | 5.01% | 3.70% | 6.49% | 8.02% | 365.0 | 1.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `full_coverage` | 100.00% | 100.00% | -0.00148 | 5.00% | 3.69% | 6.67% | 8.05% | 364.0 | 1.0 |
| `plain_global_weighted_mild_tail35_riskaux_w10` | `risk_q100` | 99.98% | 100.00% | -0.00147 | 5.00% | 3.69% | 6.67% | 8.05% | 364.0 | 1.0 |

## Read

- If RiskAux beats the post-hoc input-noise/disagreement scores at similar coverage, the learned confidence head is worth promoting.
- If it only matches post-hoc filters, the next improvement should focus on cleaner input features and stronger calibration rather than more heads.