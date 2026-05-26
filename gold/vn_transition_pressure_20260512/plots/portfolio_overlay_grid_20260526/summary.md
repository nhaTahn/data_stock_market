# Portfolio Overlay Grid

Protocol: applied on existing fold-averaged daily returns. Holdout/test not used.
Base policies tested: 197. Overlays per policy: 13.

## Top Improved Candidates (sharpe > baseline, max_dd > -0.25)

| policy                               | label                |   sharpe |   max_dd |   final_equity |   cagr |   base_sharpe |   base_dd |
|:-------------------------------------|:---------------------|---------:|---------:|---------------:|-------:|--------------:|----------:|
| full_none_r10_k10_m1                 | dd_stop_t8_p5        |   3.2506 |  -0.1201 |         5.2946 | 0.8868 |        1.1577 |   -0.4954 |
| full_none_r20_k10_m1                 | dd_vol_t8_p5_tv7     |   3.1881 |  -0.0579 |         2.383  | 0.3852 |        1.1509 |   -0.4958 |
| full_none_r10_k10_m1                 | dd_vol_t8_p5_tv7     |   3.1453 |  -0.0527 |         2.313  | 0.3693 |        1.1577 |   -0.4954 |
| full_none_r20_k10_m1                 | dd_stop_t8_p5        |   3.0798 |  -0.1445 |         4.6754 | 0.7981 |        1.1509 |   -0.4958 |
| full_none_r20_k10_m1                 | dd_vol_t10_p10_tv10  |   3.0111 |  -0.0569 |         2.7457 | 0.4633 |        1.1509 |   -0.4958 |
| conf_ratio_q70_none_r20_k10_m1       | dd_vol_t8_p5_tv7     |   2.9814 |  -0.0669 |         2.1205 | 0.3296 |        1.3125 |   -0.3631 |
| full_none_r10_k10_m1                 | dd_stop_t10_p10      |   2.8607 |  -0.1184 |         4.4491 | 0.7639 |        1.1577 |   -0.4954 |
| conf_ratio_q70_none_r20_k10_m1       | dd_stop_t8_p5        |   2.7994 |  -0.1537 |         3.7284 | 0.6543 |        1.3125 |   -0.3631 |
| full_none_r20_k10_m1                 | dd_stop_t10_p10      |   2.7281 |  -0.1149 |         4.2995 | 0.7407 |        1.1509 |   -0.4958 |
| full_none_r10_k10_m1                 | dd_vol_t10_p10_tv10  |   2.6906 |  -0.1112 |         2.4666 | 0.4038 |        1.1577 |   -0.4954 |
| full_none_r10_k10_m1                 | regime_gate_lb10_m55 |   2.6629 |  -0.1884 |         3.8163 | 0.6794 |        1.1577 |   -0.4954 |
| conf_ratio_q70_none_r20_k10_m1       | dd_vol_t10_p10_tv10  |   2.6177 |  -0.0706 |         2.2265 | 0.355  |        1.3125 |   -0.3631 |
| abs_mu_q60_pressure_nonneg_r10_k8_m1 | dd_stop_t8_p5        |   2.5794 |  -0.0719 |         3.1973 | 0.5682 |        1.3443 |   -0.2455 |
| daily_bot_sig_50pct_none_r20_k10_m1  | dd_stop_t8_p5        |   2.5704 |  -0.0883 |         2.9008 | 0.5006 |        1.1434 |   -0.3739 |
| abs_mu_q60_pressure_nonneg_r10_k8_m5 | dd_stop_t8_p5        |   2.5448 |  -0.07   |         3.1346 | 0.5562 |        1.3067 |   -0.2455 |
| abs_mu_q60_pressure_nonneg_r10_k8_m3 | dd_stop_t8_p5        |   2.5395 |  -0.0719 |         3.1281 | 0.555  |        1.3107 |   -0.2455 |
| full_none_r10_k10_m1                 | regime_gate_lb15_m55 |   2.5308 |  -0.1824 |         3.5582 | 0.6345 |        1.1577 |   -0.4954 |
| abs_mu_q60_pressure_nonneg_r10_k8_m1 | dd_vol_t8_p5_tv7     |   2.461  |  -0.0432 |         1.8053 | 0.2569 |        1.3443 |   -0.2455 |
| abs_mu_q60_pressure_nonneg_r10_k8_m5 | dd_vol_t8_p5_tv7     |   2.4517 |  -0.0412 |         1.8861 | 0.2784 |        1.3067 |   -0.2455 |
| abs_mu_q60_pressure_nonneg_r10_k8_m3 | dd_vol_t8_p5_tv7     |   2.4072 |  -0.0432 |         1.8594 | 0.2714 |        1.3107 |   -0.2455 |

## Policy Overlay Summary (best overlay per policy)

| policy                                |   base_sharpe |   base_max_dd |   base_final_equity | best_overlay         |   best_sharpe |   best_max_dd |   best_final_equity |   sharpe_delta |   dd_delta |
|:--------------------------------------|--------------:|--------------:|--------------------:|:---------------------|--------------:|--------------:|--------------------:|---------------:|-----------:|
| full_none_r10_k10_m1                  |        1.1577 |       -0.4954 |              2.3361 | dd_stop_t8_p5        |        3.2506 |       -0.1201 |              5.2946 |         2.0929 |     0.3752 |
| full_none_r20_k10_m1                  |        1.1509 |       -0.4958 |              2.3086 | dd_vol_t8_p5_tv7     |        3.1881 |       -0.0579 |              2.383  |         2.0372 |     0.4379 |
| conf_ratio_q70_none_r20_k10_m1        |        1.3125 |       -0.3631 |              2.4214 | dd_vol_t8_p5_tv7     |        2.9814 |       -0.0669 |              2.1205 |         1.6689 |     0.2963 |
| abs_mu_q60_pressure_nonneg_r10_k8_m1  |        1.3443 |       -0.2455 |              2.1283 | dd_stop_t8_p5        |        2.5794 |       -0.0719 |              3.1973 |         1.2351 |     0.1736 |
| daily_bot_sig_50pct_none_r20_k10_m1   |        1.1434 |       -0.3739 |              1.9465 | dd_stop_t8_p5        |        2.5704 |       -0.0883 |              2.9008 |         1.427  |     0.2856 |
| abs_mu_q60_pressure_nonneg_r10_k8_m5  |        1.3067 |       -0.2455 |              2.076  | dd_stop_t8_p5        |        2.5448 |       -0.07   |              3.1346 |         1.238  |     0.1755 |
| abs_mu_q60_pressure_nonneg_r10_k8_m3  |        1.3107 |       -0.2455 |              2.0822 | dd_stop_t8_p5        |        2.5395 |       -0.0719 |              3.1281 |         1.2288 |     0.1736 |
| abs_mu_q60_pressure_nonneg_r20_k8_m1  |        1.3049 |       -0.2455 |              2.1203 | dd_stop_t8_p5        |        2.3883 |       -0.0836 |              2.8778 |         1.0833 |     0.1619 |
| abs_mu_q60_pressure_nonneg_r20_k8_m5  |        1.2914 |       -0.2455 |              2.0987 | dd_stop_t8_p5        |        2.3852 |       -0.0817 |              2.8629 |         1.0938 |     0.1638 |
| abs_mu_q60_pressure_nonneg_r20_k8_m3  |        1.2954 |       -0.2455 |              2.105  | dd_stop_t8_p5        |        2.3799 |       -0.0836 |              2.8569 |         1.0845 |     0.1619 |
| abs_mu_q60_pressure_nonneg_r10_k15_m1 |        1.2669 |       -0.2563 |              1.9732 | dd_vol_t8_p5_tv7     |        2.3536 |       -0.0718 |              1.7384 |         1.0867 |     0.1846 |
| abs_mu_q60_pressure_nonneg_r20_k10_m6 |        1.3176 |       -0.2547 |              2.0905 | dd_vol_t8_p5_tv7     |        2.3331 |       -0.0534 |              1.7947 |         1.0155 |     0.2013 |
| abs_mu_q60_pressure_nonneg_r10_k15_m3 |        1.2318 |       -0.2563 |              1.9305 | dd_vol_t8_p5_tv7     |        2.3309 |       -0.0718 |              1.723  |         1.0991 |     0.1846 |
| abs_mu_q60_pressure_nonneg_r20_k5_m5  |        1.1843 |       -0.2319 |              2.0001 | dd_vol_t8_p5_tv7     |        2.3226 |       -0.0564 |              1.7433 |         1.1383 |     0.1755 |
| daily_bot_sig_50pct_none_r10_k10_m1   |        0.8541 |       -0.4086 |              1.6059 | dd_stop_t8_p5        |        2.3028 |       -0.0774 |              2.5642 |         1.4487 |     0.3312 |
| full_pressure_nonneg_r20_k5_m5        |        1.2009 |       -0.2319 |              2.0241 | dd_vol_t8_p5_tv7     |        2.3028 |       -0.0564 |              1.7348 |         1.1019 |     0.1755 |
| full_pressure_nonneg_r20_k5_m6        |        1.2009 |       -0.2319 |              2.0241 | dd_vol_t8_p5_tv7     |        2.3028 |       -0.0564 |              1.7348 |         1.1019 |     0.1755 |
| full_pressure_nonneg_r20_k5_m3        |        1.2009 |       -0.2319 |              2.0241 | dd_vol_t8_p5_tv7     |        2.3028 |       -0.0564 |              1.7348 |         1.1019 |     0.1755 |
| full_pressure_nonneg_r20_k5_m1        |        1.2009 |       -0.2319 |              2.0241 | dd_vol_t8_p5_tv7     |        2.3028 |       -0.0564 |              1.7348 |         1.1019 |     0.1755 |
| conf_ratio_q70_none_r10_k10_m1        |        0.728  |       -0.445  |              1.5495 | regime_gate_lb10_m55 |        2.3007 |       -0.1592 |              2.7656 |         1.5727 |     0.2858 |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/portfolio_overlay_grid_20260526",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/portfolio_overlay_grid_20260526"
}