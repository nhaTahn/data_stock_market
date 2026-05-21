# Market Leader Signal Grid

Feature hypothesis: replace the old hard-coded `vingroup_momentum` with a causal basket of market leaders.

Selection rule: choose `top_k`, liquidity window, and raw/excess variant by train rel_score improvement over a one-factor market-return baseline; report validation separately.

## Selected By Train

- `top_k`: `3`
- `liquidity_window`: `60`
- `min_periods`: `20`
- `selected_by_train_signal_column`: `market_leader_return_k3_w60`
- `selected_signal_variant`: `raw`
- `selected_train_rel_improvement`: `0.010368751819968991`
- `selected_val_rel_improvement`: `0.001435106935760655`
- `selected_val_spearman_ic`: `0.018258005191786227`
- `selected_val_sign_accuracy`: `0.5417298937784522`
- `selected_val_timing_equity_net`: `0.8262670616826714`
- `selected_val_timing_t_stat_net`: `-0.2557132153553059`

## Top Validation Rows

|   top_k |   liquidity_window | selected_by_train_signal_column      | selected_signal_variant   |   selected_train_rel_improvement |   selected_val_rel_improvement |   selected_val_spearman_ic |   selected_val_sign_accuracy |   selected_val_timing_equity_net |   selected_val_timing_t_stat_net |
|--------:|-------------------:|:-------------------------------------|:--------------------------|---------------------------------:|-------------------------------:|---------------------------:|-----------------------------:|---------------------------------:|---------------------------------:|
|      25 |                 90 | market_leader_return_k25_w90         | raw                       |                       0.00453277 |                     0.00491521 |                  0.0302476 |                     0.550835 |                         0.782149 |                       -0.387884  |
|      30 |                 90 | market_leader_return_k30_w90         | raw                       |                       0.00508223 |                     0.00426585 |                  0.0302323 |                     0.552352 |                         0.85909  |                       -0.161881  |
|      30 |                120 | market_leader_return_k30_w120_excess | excess                    |                       0.00488585 |                     0.00377123 |                 -0.0462817 |                     0.459788 |                         0.293774 |                       -2.73598   |
|      20 |                 60 | market_leader_return_k20_w60         | raw                       |                       0.00698334 |                     0.00364566 |                  0.0262111 |                     0.5478   |                         0.808982 |                       -0.30675   |
|      25 |                120 | market_leader_return_k25_w120        | raw                       |                       0.00615427 |                     0.00317369 |                  0.0321992 |                     0.552352 |                         0.78515  |                       -0.378321  |
|      25 |                 60 | market_leader_return_k25_w60         | raw                       |                       0.00730352 |                     0.0031198  |                  0.0314106 |                     0.552352 |                         0.886981 |                       -0.0846907 |
|      20 |                 20 | market_leader_return_k20_w20_excess  | excess                    |                       0.00805052 |                     0.00304141 |                 -0.0610829 |                     0.456753 |                         0.21858  |                       -3.4597    |
|      12 |                 60 | market_leader_return_k12_w60         | raw                       |                       0.00910109 |                     0.00240082 |                  0.0137718 |                     0.53566  |                         0.823287 |                       -0.263946  |
|      15 |                 90 | market_leader_return_k15_w90         | raw                       |                       0.00661018 |                     0.00232912 |                  0.0282066 |                     0.54173  |                         0.799649 |                       -0.332699  |
|      12 |                 90 | market_leader_return_k12_w90         | raw                       |                       0.00724404 |                     0.00229151 |                  0.023288  |                     0.538695 |                         0.738695 |                       -0.525036  |

## Latest Leaders For Selected Config

| Date                |   leader_rank | code   |   liquidity_score |   stock_return_1 |
|:--------------------|--------------:|:-------|------------------:|-----------------:|
| 2026-03-31 00:00:00 |             1 | HPG    |       1.27049e+12 |       0          |
| 2026-03-31 00:00:00 |             2 | FPT    |       1.14469e+12 |       0.00945946 |
| 2026-03-31 00:00:00 |             3 | SSI    |       1.08637e+12 |       0.0150659  |
