# VN Reporting Standard

This document defines the default reporting preset that is now enforced by the training and report-refresh flow.

## Market And Time

- `market = VN`
- `target_mode = return`
- `train = [..., 31/03/2020]`
- `in_sample = [01/04/2020, 15/11/2022]`
- `out_sample = [16/11/2022, now]`

Internally the pipeline still stores the split keys as `train`, `val`, and `test`, but reports expose them as:

- `train`
- `in_sample`
- `out_sample`

## Evaluation Standard

Each report refresh now recomputes:

- `rel_score`
- `rmse`
- `directional_accuracy`
- error histogram with `q2` and `q8` (`0.2` and `0.8` quantiles)
- rel_score histogram
- quartile trading curve: long top `1/4`, short bottom `1/4`
- large-error extract

By default, these artifacts are written only for `train` and `in_sample`. `out_sample` is a hidden holdout and should only be revealed for a final one-time check.

## Enforced Defaults

`python3 main.py train` now defaults to:

- `--market VN`
- `--target-mode return`
- `--train-end-date 2020-03-31`
- `--val-end-date 2022-11-15`

If a run tries to change the VN split, the command now fails unless `--allow-nonstandard-time` is passed explicitly.

## Generated Artifacts

For each refreshed run, the reporting layer writes:

- `reports/core/reporting_standard.json`
- `reports/core/evaluation_summary.json`
- `reports/core/feature_formula_report.md`
- `reports/core/large_error_<model>_<split>.csv`
- `reports/plots/error_hist_<model>_<split>.png`
- `reports/plots/rel_score_hist_<model>.png`
- `reports/backtests/quartile_long_short_<model>_<split>.csv`
- `reports/backtests/quartile_long_short_equity_<model>_<split>.png`

## Commands

Refresh a saved run:

```bash
python3 main.py report update-run /path/to/run_dir
```

Reveal the hidden out-sample holdout only for the final check:

```bash
python3 main.py report update-run /path/to/run_dir --reveal-out-sample
```

Write the feature formula reference:

```bash
python3 main.py report feature-formulas
```

Train with the enforced VN preset:

```bash
python3 main.py train
```
