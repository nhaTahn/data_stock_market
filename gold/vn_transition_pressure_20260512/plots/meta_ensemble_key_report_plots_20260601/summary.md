# Meta-Ensemble Key Report Plots

Scope: VN validation only. Holdout/test is not used.
Model: `hgb_abs_blend` meta-ensemble over frozen 5-seed hetero anchor.

Overall validation rel_score: `0.048809`.
Blend alpha: `0.475`.

## Daily Error Summary

| series           |   median |      p90 |      max |   days_gt_3pct |   days_gt_3p5pct |
|:-----------------|---------:|---------:|---------:|---------------:|-----------------:|
| median_abs_error | 0.011646 | 0.024806 | 0.080161 |             47 |               30 |
| q75_abs_error    | 0.021976 | 0.044818 | 0.095388 |            160 |              119 |
| q90_abs_error    | 0.036383 | 0.062609 | 0.115699 |            469 |              360 |

## rel_score by Code Summary

|       |   rel_score |
|:------|------------:|
| count |   93        |
| mean  |    0.045649 |
| std   |    0.034449 |
| min   |   -0.040627 |
| 25%   |    0.026363 |
| 50%   |    0.044432 |
| 75%   |    0.072195 |
| max   |    0.129837 |

## Files

- `rel_score_histogram_by_code.png`
- `daily_return_error_timeseries.png`
- `rel_score_by_code.csv`
- `daily_return_error_series.csv`
- `validation_predictions.csv`

{
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/meta_ensemble_key_report_plots_20260601",
  "report_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_key_report_plots_20260601"
}