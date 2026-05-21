# Teacher-Style Current Best Error-Control Plot

Scope: validation accepted samples only. Holdout/test is not used.
Policy: `risk_hgb/coverage_q40`.
Index line: `VN100`.
Error universe: `all`.
Seeds aggregated by daily mean q90: `43, 52, 71`.

Formula:

```text
E_d = { actual_return_{i,d} - predicted_return_{i,d} } for accepted stocks
seed_error_s(d) = Q_0.90(|E_d|) within each seed s
ts_error(d) = mean_s seed_error_s(d)
```

- Days: `236`
- Date range: `2020-06-17` to `2022-09-21`
- Median accepted stocks/seed/day: `28.8`
- Median q90(|E|): `0.02112`
- P90 q90(|E|): `0.03374`
- Max q90(|E|): `0.06269`

Files:

- `teacher_style_index_vs_q90_abs_error.png`
- `teacher_style_index_vs_q90_abs_error_by_year.png`
- `teacher_style_abs_error.csv`
- `by_year/vn100_accepted_q90_abs_error_2020.png`
- `by_year/vn100_accepted_q90_abs_error_2021.png`
- `by_year/vn100_accepted_q90_abs_error_2022.png`