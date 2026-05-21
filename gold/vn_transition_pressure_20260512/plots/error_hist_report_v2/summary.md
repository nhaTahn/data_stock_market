# VN Gold Error Histogram Report

Scope: train and in-sample validation only. Holdout/test is not used.

E is defined as:

```text
E = prediction - actual
```

## Summary

| Candidate | Split | Stocks | Rows | rel_score | base_score | abs_score | dir_acc | q20 | q80 | mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_lstm` | `train` | 92 | 128421 | -0.00454 | 0.02822 | 0.02834 | 42.71% | -0.01326 | +0.01362 | +0.00009 |
| `base_lstm` | `in_sample` | 93 | 59259 | +0.00485 | 0.03738 | 0.03720 | 47.10% | -0.01733 | +0.01620 | +0.00017 |
| `base_lstm` | `train_in_sample` | 93 | 187680 | -0.00112 | 0.03091 | 0.03094 | 44.91% | -0.01451 | +0.01433 | +0.00011 |
| `lstm_filter_move_top_train_ic` | `train` | 92 | 128421 | -0.00276 | 0.02822 | 0.02829 | 28.04% | -0.01325 | +0.01366 | +0.00013 |
| `lstm_filter_move_top_train_ic` | `in_sample` | 93 | 59259 | +0.00767 | 0.03738 | 0.03709 | 24.37% | -0.01745 | +0.01600 | +0.00003 |
| `lstm_filter_move_top_train_ic` | `train_in_sample` | 93 | 187680 | -0.00049 | 0.03091 | 0.03092 | 26.20% | -0.01449 | +0.01431 | +0.00010 |

## Plots

- `base_lstm` `train`: `error_hist_base_lstm_train.png`
- `base_lstm` `in_sample`: `error_hist_base_lstm_in_sample.png`
- `base_lstm` `train_in_sample`: `error_hist_base_lstm_train_in_sample.png`
- `lstm_filter_move_top_train_ic` `train`: `error_hist_lstm_filter_move_top_train_ic_train.png`
- `lstm_filter_move_top_train_ic` `in_sample`: `error_hist_lstm_filter_move_top_train_ic_in_sample.png`
- `lstm_filter_move_top_train_ic` `train_in_sample`: `error_hist_lstm_filter_move_top_train_ic_train_in_sample.png`
