# Teacher-Style Absolute Error Plot

Scope: `train` split only. Holdout/test is not used.
Index line: `VN30`.
Error universe: `same_as_index`.

Formula:

```text
E_d = { actual_return_{i,d} - predicted_return_{i,d} }
ts_error(d) = Q_0.90(|E_d|)
```

The plot intentionally keeps only two lines: index proxy and q90 absolute return error.

- Days: `1959`
- Date range: `2012-05-28` to `2020-03-31`
- Median stocks/day used in error: `16`
- Median q90(|E|): `0.02435`

Files:

- `teacher_style_index_vs_q90_abs_error.png`
- `teacher_style_index_vs_q90_abs_error_by_year.png`
- `by_year/*.png`
- `teacher_style_abs_error.csv`
