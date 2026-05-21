# Portable LSTM Filter Signal Stability

Scope: train/validation only. Holdout/test data was not opened.

## Runs Compared

| Role | Report | Base model | Best validation candidate | Validation rel_score | Coverage | Mean daily IC | IC t |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Primary | `portable_lstm_filter_signal_20260508_r04` | `lstm_seed_52` | `move_top_train_ic_selected` | +0.00717 | 40.5% | +0.02376 | +3.05 |
| Refactor verification | `portable_lstm_filter_signal_20260509_r06_selector_module` | `lstm_seed_52` | `move_top_train_ic_selected` | +0.00717 | 40.5% | +0.02376 | +3.05 |
| Challenger check | `portable_lstm_filter_signal_20260508_r05_signmag` | `lstm_signmag_seed_52` | `daily_top_20` | +0.00561 | 20.5% | +0.01135 | +1.69 |
| Challenger check | `portable_lstm_filter_signal_20260508_r05_signmag` | `lstm_signmag_seed_52` | `gate` | +0.00556 | 2.8% | +0.02932 | +1.21 |

## Read

- The filter layer is not a one-off artifact: it improves validation `rel_score` on both plain LSTM and signmag base streams.
- The strongest current path remains plain `lstm_seed_52` plus expected-move daily selection: `abs(base_prediction) * filter_probability`, with coverage selected on train mean daily IC.
- The reusable selector module was full-run verified in `portable_lstm_filter_signal_20260509_r06_selector_module`; metrics and selected params match the pre-refactor `r04` exactly.
- Signmag confirms the filter idea, but its best validation result is weaker than the plain LSTM filter path.
- Naive risk-scaled output remains rejected. For both base streams, risk scaling hurts `rel_score`; keep market/regime variables as filter inputs until a separate calibration is trained.

## Next Engineering Step

The best selector has been converted into a reusable module:

- module: `src/models/selection/filter_signal.py`
- fit API: `fit_filter_signal_selection()`
- apply API: `apply_filter_signal_selection()`

The production-facing contract is:

- input: base predictions, filter probabilities, market proxy context
- score: `abs(base_prediction) * filter_probability`
- selector: daily top coverage selected on train mean daily IC
- output: selected prediction stream plus coverage diagnostics
