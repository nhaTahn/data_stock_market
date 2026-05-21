# Quick Gold Report

Source run: `data/processed/assets/data_info_vn/history/training_runs/vin_h3_p5_overnight_pcie_lite_phase1_20260417`
Model: `lstm_attention`
Stocks: VHM, VIC, VRE

This package uses the **test split** only for reporting.
`E` is defined as `prediction - actual` after the same per-stock temporal alignment used by `rel_score` evaluation.

## Scores

- `rel_score`: 0.043404
- `base_score`: 0.047473
- `abs_score`: 0.045413
- `raw_test_rows`: 868
- `aligned_error_rows`: 862

## Error Summary

- `min`: -0.078025
- `q25`: -0.015024
- `median`: 0.001245
- `q75`: 0.014165
- `max`: 0.084492
- `mean`: 0.000144
- `std`: 0.031628

Main artifact: `plots/error_histogram_E_prediction_minus_actual.png`
