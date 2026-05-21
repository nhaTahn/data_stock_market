# LSTM Calibration Multiseed Readout

Calibration is fitted on the train split and evaluated on validation. Holdout/test is not used.

## Validation By Seed

|   seed | candidate                     |   rel_score | q90_abs_error   | daily_q90_abs_error_q90   | daily_q90_abs_error_max   |   spike_days_ge_7pct |   spike_days_ge_8pct |   prediction_actual_abs_q90_ratio |
|-------:|:------------------------------|------------:|:----------------|:--------------------------|:--------------------------|---------------------:|---------------------:|----------------------------------:|
|     43 | weighted_scale                |     0.03057 | 4.822%          | 6.450%                    | 10.688%                   |                   40 |                   12 |                           0.15701 |
|     43 | weighted                      |     0.02968 | 4.831%          | 6.460%                    | 10.508%                   |                   40 |                   10 |                           0.14953 |
|     43 | bw_tail_penalty_0p25          |     0.02954 | 4.841%          | 6.450%                    | 10.248%                   |                   38 |                    9 |                           0.14174 |
|     43 | simplex_tail_penalty_0p25     |     0.02954 | 4.841%          | 6.450%                    | 10.248%                   |                   38 |                    9 |                           0.14174 |
|     43 | simplex_tail_penalty_0p50     |     0.02954 | 4.841%          | 6.450%                    | 10.248%                   |                   38 |                    9 |                           0.14174 |
|     43 | base_weighted_ensemble        |     0.02621 | 4.862%          | 6.533%                    | 9.731%                    |                   35 |                   10 |                           0.13442 |
|     43 | simplex_ensemble              |     0.01784 | 4.923%          | 6.503%                    | 9.247%                    |                   36 |                    8 |                           0.12151 |
|     43 | base                          |     0.01719 | 4.918%          | 6.497%                    | 9.361%                    |                   39 |                   14 |                           0.14206 |
|     43 | weighted_instance_tail_switch |     0.01304 | 4.954%          | 6.550%                    | 8.165%                    |                   36 |                    2 |                           0.11662 |
|     43 | instance                      |     0.00448 | 5.010%          | 6.571%                    | 8.183%                    |                   36 |                    2 |                           0.11649 |
|     52 | weighted_scale                |     0.02731 | 4.828%          | 6.372%                    | 12.723%                   |                   44 |                   17 |                           0.16361 |
|     52 | weighted                      |     0.02705 | 4.835%          | 6.390%                    | 12.188%                   |                   41 |                   17 |                           0.14873 |
|     52 | simplex_ensemble              |     0.02683 | 4.844%          | 6.424%                    | 11.504%                   |                   38 |                   16 |                           0.13685 |
|     52 | bw_tail_penalty_0p25          |     0.0265  | 4.852%          | 6.497%                    | 11.289%                   |                   35 |                   16 |                           0.13287 |
|     52 | simplex_tail_penalty_0p25     |     0.0265  | 4.852%          | 6.497%                    | 11.289%                   |                   35 |                   16 |                           0.13287 |
|     52 | simplex_tail_penalty_0p50     |     0.0265  | 4.852%          | 6.497%                    | 11.289%                   |                   35 |                   16 |                           0.13287 |
|     52 | base_weighted_ensemble        |     0.02641 | 4.850%          | 6.454%                    | 11.382%                   |                   36 |                   16 |                           0.13509 |
|     52 | base                          |     0.01539 | 4.934%          | 6.585%                    | 9.624%                    |                   38 |                    9 |                           0.13606 |
|     52 | weighted_instance_tail_switch |     0.01057 | 4.951%          | 6.552%                    | 8.543%                    |                   39 |                    1 |                           0.11295 |
|     52 | instance                      |     0.00613 | 4.989%          | 6.670%                    | 8.543%                    |                   37 |                    1 |                           0.09492 |
|     71 | base                          |     0.01744 | 4.927%          | 6.606%                    | 9.534%                    |                   44 |                   16 |                           0.12324 |
|     71 | bw_tail_penalty_0p25          |     0.01719 | 4.929%          | 6.623%                    | 9.279%                    |                   42 |                   14 |                           0.11035 |
|     71 | base_weighted_ensemble        |     0.01681 | 4.931%          | 6.618%                    | 9.364%                    |                   44 |                   14 |                           0.11461 |
|     71 | simplex_tail_penalty_0p25     |     0.01403 | 4.965%          | 6.598%                    | 8.857%                    |                   40 |                    4 |                           0.11778 |
|     71 | simplex_tail_penalty_0p50     |     0.01403 | 4.965%          | 6.598%                    | 8.857%                    |                   40 |                    4 |                           0.11778 |
|     71 | simplex_ensemble              |     0.01088 | 4.982%          | 6.552%                    | 8.714%                    |                   35 |                    3 |                           0.12731 |
|     71 | instance                      |     0.00889 | 4.990%          | 6.554%                    | 8.615%                    |                   35 |                    3 |                           0.13323 |
|     71 | weighted_instance_tail_switch |     0.0076  | 4.973%          | 6.611%                    | 8.615%                    |                   35 |                    3 |                           0.1066  |
|     71 | weighted                      |     0.00391 | 4.974%          | 6.523%                    | 7.992%                    |                   45 |                    0 |                           0.09081 |
|     71 | weighted_scale                |     0.00231 | 5.004%          | 6.730%                    | 7.251%                    |                   32 |                    0 |                           0.02724 |

## Validation Aggregate

| candidate                     |   rel_score_mean |   rel_score_std | q90_abs_error_mean   | daily_q90_abs_error_q90_mean   | daily_q90_abs_error_max_mean   |   spike_days_ge_7pct_mean |   spike_days_ge_8pct_mean |   prediction_actual_abs_q90_ratio_mean |
|:------------------------------|-----------------:|----------------:|:---------------------|:-------------------------------|:-------------------------------|--------------------------:|--------------------------:|---------------------------------------:|
| bw_tail_penalty_0p25          |          0.02441 |         0.00644 | 4.874%               | 6.523%                         | 10.272%                        |                   38.3333 |                  13       |                                0.12832 |
| simplex_tail_penalty_0p25     |          0.02336 |         0.00822 | 4.886%               | 6.515%                         | 10.132%                        |                   37.6667 |                   9.66667 |                                0.1308  |
| simplex_tail_penalty_0p50     |          0.02336 |         0.00822 | 4.886%               | 6.515%                         | 10.132%                        |                   37.6667 |                   9.66667 |                                0.1308  |
| base_weighted_ensemble        |          0.02314 |         0.00548 | 4.881%               | 6.535%                         | 10.159%                        |                   38.3333 |                  13.3333  |                                0.12804 |
| weighted                      |          0.02021 |         0.01419 | 4.880%               | 6.458%                         | 10.229%                        |                   42      |                   9       |                                0.12969 |
| weighted_scale                |          0.02006 |         0.01546 | 4.885%               | 6.517%                         | 10.221%                        |                   38.6667 |                   9.66667 |                                0.11595 |
| simplex_ensemble              |          0.01852 |         0.008   | 4.916%               | 6.493%                         | 9.821%                         |                   36.3333 |                   9       |                                0.12856 |
| base                          |          0.01667 |         0.00112 | 4.926%               | 6.563%                         | 9.506%                         |                   40.3333 |                  13       |                                0.13379 |
| weighted_instance_tail_switch |          0.0104  |         0.00273 | 4.959%               | 6.571%                         | 8.441%                         |                   36.6667 |                   2       |                                0.11206 |
| instance                      |          0.0065  |         0.00223 | 4.997%               | 6.598%                         | 8.447%                         |                   36      |                   2       |                                0.11488 |

## Fitted Parameters

|   seed |   weighted_scale |   bw_base |   bw_weighted |   simplex_base |   simplex_weighted |   simplex_instance |   switch_threshold |   bw_tail_0p25_base |   bw_tail_0p25_weighted |   simplex_tail_0p25_base |   simplex_tail_0p25_weighted |   simplex_tail_0p25_instance |   simplex_tail_0p50_base |   simplex_tail_0p50_weighted |   simplex_tail_0p50_instance |
|-------:|-----------------:|----------:|--------------:|---------------:|-------------------:|-------------------:|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------------:|-----------------------------:|-------------------------:|-----------------------------:|-----------------------------:|
|     43 |             1.05 |      0.55 |          0.45 |            0.6 |                0.2 |                0.2 |         0.00868976 |                0.2  |                    0.8  |                      0.2 |                          0.8 |                  0           |                      0.2 |                          0.8 |                  0           |
|     52 |             1.1  |      0.25 |          0.75 |            0.2 |                0.8 |                0   |         0.00823238 |                0.3  |                    0.7  |                      0.3 |                          0.7 |                 -1.11022e-16 |                      0.3 |                          0.7 |                 -1.11022e-16 |
|     71 |             0.3  |      0.9  |          0.1  |            0.1 |                0   |                0.9 |         0.00155748 |                0.85 |                    0.15 |                      0.3 |                          0   |                  0.7         |                      0.3 |                          0   |                  0.7         |

## Read

- A calibrated/ensemble prediction is useful only if it improves `rel_score` without increasing daily tail spikes.
- If the best train-fitted calibration is close to `weighted`, the model improvement comes from mild imbalance weighting.
- If the best train-fitted calibration adds `instance`, the spike-control signal is useful but should be promoted carefully because it can under-amplify returns.