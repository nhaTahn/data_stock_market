# Frozen Validation Candidate Advisor Report

Protocol: train `<= 2020-03-31`, validation/in-sample `2020-04-01..2022-11-15`. Holdout/test not used.

## Recommendation

- Prediction candidate: `ensemble_mean_cal_each_traincal_clip` from cached `hetero_combined_full5_20260521`.
- Portfolio/risk overlay: `daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5` with gate `wyck040`.
- Keep holdout closed until these choices are frozen and no further validation tuning is planned.

## Prediction Target Metrics

| Metric | Frozen ensemble | Prior fixed train_cal reference |
| --- | ---: | ---: |
| rel_score | **0.04478** | 0.03493 |
| absE_robust | **3.60%** | 3.64% |
| absE_q90 | **4.69%** | 4.76% |
| DA | **51.83%** | 50.97% |
| pred/actual q90 ratio | **0.193** | 0.184 |

## 21-Day Fold Stability

| Metric | Value |
| --- | ---: |
| mean fold rel_score | 0.02980 |
| median fold rel_score | 0.02077 |
| minimum fold rel_score | -0.02823 |
| positive folds | 24/32 |
| mean absE_robust | 3.64% |
| p90 absE_robust | 5.01% |

## Yearly Prediction Series

|   year |   days |   mean_rel_score |   median_rel_score |   positive_days |   mean_absE_robust |   p90_absE_robust |   mean_absE_q90 |   mean_DA |
|-------:|-------:|-----------------:|-------------------:|----------------:|-------------------:|------------------:|----------------:|----------:|
|   2020 |    193 |          0.01254 |            0.00779 |             103 |            0.02915 |           0.04402 |         0.03476 |   0.50612 |
|   2021 |    250 |          0.01448 |            0.01351 |             143 |            0.03348 |           0.04753 |         0.03907 |   0.52233 |
|   2022 |    216 |          0.00586 |            0.00477 |             116 |            0.04198 |           0.06773 |         0.04696 |   0.52517 |

## Worst Prediction Folds

| test_start   | test_end   |    n |   rel_score |   absE_robust |   absE_q90 |      DA |   pred_actual_q90_ratio |
|:-------------|:-----------|-----:|------------:|--------------:|-----------:|--------:|------------------------:|
| 2021-03-04   | 2021-04-01 | 1922 |    -0.02823 |       0.02631 |    0.0329  | 0.51301 |                 0.17847 |
| 2020-10-29   | 2020-11-26 | 1904 |    -0.02448 |       0.02577 |    0.03296 | 0.46324 |                 0.16963 |
| 2022-08-05   | 2022-09-06 | 1949 |    -0.01766 |       0.02646 |    0.03298 | 0.47614 |                 0.15865 |
| 2022-07-07   | 2022-08-04 | 1943 |    -0.01605 |       0.03285 |    0.04156 | 0.46835 |                 0.17689 |
| 2020-09-30   | 2020-10-28 | 1920 |    -0.01256 |       0.02488 |    0.03096 | 0.47031 |                 0.15789 |
| 2022-02-07   | 2022-03-07 | 1941 |    -0.00944 |       0.03162 |    0.03939 | 0.48789 |                 0.14917 |
| 2021-06-04   | 2021-07-02 | 1920 |    -0.00524 |       0.03216 |    0.04078 | 0.48385 |                 0.1496  |
| 2020-08-31   | 2020-09-29 | 1920 |    -0.00243 |       0.02136 |    0.02687 | 0.46094 |                 0.15209 |
| 2020-12-28   | 2021-01-26 | 1913 |     0.0021  |       0.04252 |    0.05752 | 0.52901 |                 0.12454 |
| 2021-05-06   | 2021-06-03 | 1915 |     0.00552 |       0.03373 |    0.04126 | 0.53577 |                 0.14113 |
| 2021-10-04   | 2021-11-01 | 1946 |     0.00798 |       0.02578 |    0.03311 | 0.51182 |                 0.14895 |
| 2020-07-31   | 2020-08-28 | 1908 |     0.00888 |       0.02441 |    0.03044 | 0.50524 |                 0.18949 |

## Portfolio Overlay

| Metric | Value |
| --- | ---: |
| mean equity | 1.5231 |
| min seed equity | 1.4417 |
| mean annual return | 17.67% |
| mean Sharpe | 1.2436 |
| worst max drawdown | -18.75% |
| worst fold equity | 0.9532 |
| positive folds | 77/155 |
| gate active days | 59.48% |

## Portfolio Seed Metrics

| policy                                         | gate    |   seed |   active_days |   final_equity |   ann_ret |   sharpe |   max_dd |   hit_rate |
|:-----------------------------------------------|:--------|-------:|--------------:|---------------:|----------:|---------:|---------:|-----------:|
| daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 | wyck040 |     43 |       0.60215 |        1.56867 |   0.19039 |  1.33727 | -0.16313 |    0.32412 |
| daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 | wyck040 |     52 |       0.60215 |        1.47199 |   0.16144 |  1.13937 | -0.17071 |    0.32104 |
| daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 | wyck040 |     62 |       0.60215 |        1.58664 |   0.19565 |  1.3534  | -0.17015 |    0.30876 |
| daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 | wyck040 |     71 |       0.60215 |        1.44171 |   0.15213 |  1.06627 | -0.1875  |    0.31029 |
| daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 | wyck040 |     82 |       0.60215 |        1.54628 |   0.18378 |  1.32193 | -0.17815 |    0.31797 |

## Notes

- Full-universe rel_score should be judged on the ungated prediction candidate.
- Wyckoff/pressure gates are execution overlays; they reduce crash exposure but should not be used to score full-universe prediction quality.
- Short-window rolling retrain (`w126`, and even `w504`) remains a robustness stress-test, not the main objective protocol.
- The next step before holdout is to freeze this candidate and generate the final run manifest/config from these exact artifacts.