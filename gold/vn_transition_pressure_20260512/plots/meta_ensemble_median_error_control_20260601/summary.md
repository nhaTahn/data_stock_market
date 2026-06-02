# Meta-Ensemble Median Error Control

Scope: VN validation only. Holdout/test is not used.

This report is intentionally based on daily median absolute error, not daily Q90 tail error.
The goal is to show the central-error stability of the frozen `hgb_abs_blend` meta-ensemble.

## Summary

| series                     |   days |   median |      p90 |      max |   days_gt_3pct |   share_gt_3pct |
|:---------------------------|-------:|---------:|---------:|---------:|---------------:|----------------:|
| median_abs_error           |    659 | 0.011646 | 0.024806 | 0.080161 |             47 |        0.07132  |
| mean_abs_error             |    659 | 0.016392 | 0.029816 | 0.081857 |             65 |        0.098634 |
| q75_abs_error              |    659 | 0.021976 | 0.044818 | 0.095388 |            160 |        0.242792 |
| q90_abs_error              |    659 | 0.036383 | 0.062609 | 0.115699 |            469 |        0.711684 |
| rolling20_median_abs_error |    653 | 0.01128  | 0.01872  | 0.026846 |              0 |        0        |
| rolling30_median_abs_error |    650 | 0.011873 | 0.018051 | 0.024636 |              0 |        0        |

## Read

- Daily median |E| has a low central-error level but can spike on market-shock days.
- The 20-day rolling median |E| stays below the 3.0% target throughout validation.
- Keep this separate from Q90/tail-risk plots; Q90 remains the honest tail-stress diagnostic.

{
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/meta_ensemble_median_error_control_20260601",
  "local_output": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_median_error_control_20260601"
}