# VN Next-Day Return Prediction Plots

Scope: VN train/in-sample artifact only. Holdout/test is not used.

- Latest cross-section signal date: `2022-11-14`
- Actual next-day date: `2022-11-15`
- Time-series sample codes: `VCB, FPT, HPG, SSI, VIC, VHM`

## Plots

- `base_lstm_latest_all_codes.png`: Base LSTM predicted next-day return sorted across all VN codes on the latest eligible in-sample signal date.
- `base_lstm_latest_top_bottom_prediction_actual.png`: Top/bottom Base LSTM signals with actual next-day return overlay.
- `base_lstm_selected_codes_timeseries.png`: Base LSTM prediction vs actual return over time for representative VN codes.
- `base_lstm_per_code_mean_prediction_vs_actual.png`: Per-code mean prediction vs mean actual return in-sample.

Read: this plot checks whether Base LSTM has usable forecasting signal. It is not a trading backtest; trading suitability is evaluated separately with net equity, Sharpe, drawdown, turnover, and cost.
