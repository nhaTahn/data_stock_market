# Multi-Market Framework Branch Manifest

Branch: `codex/multimarket-framework`

## Purpose

Keep the current best VN candidate intact while developing a more ambitious
multi-market research framework. This branch should stay lightweight: commit
code, compact summaries, and selected figures only; keep heavy prediction
arrays and run folders local unless explicitly promoted.

## Current Frozen VN Anchor

Do not replace this anchor unless a new candidate beats it on the same
validation protocol without using holdout.

| Component | Choice |
| --- | --- |
| Prediction model | `hetero_combined_full5 -> mean ensemble -> train calibration/clip` |
| Prediction variant | `ensemble_mean_cal_each_traincal_clip` |
| Validation protocol | train `<= 2020-03-31`, validation `2020-04-01..2022-11-15` |
| rel_score | `0.04478` |
| DA | `51.83%` |
| absE_robust | `3.60%` |
| Portfolio overlay | `daily_bot_sig_50pct + wyck040 + r20/k20/m5` |
| Holdout | Closed |

## Relevant Code To Keep

Core VN academic package:
- `experiments/training/evaluate_fixed_train_ensemble_calibration.py`
- `experiments/training/build_frozen_validation_candidate_report.py`
- `experiments/training/run_academic_ablation_baseline_significance.py`
- `experiments/training/make_report_plots.py`

Multi-market framework package:
- `experiments/training/run_multimarket_portable_baseline_significance.py`
- `experiments/training/evaluate_multimarket_portable_ensemble.py`
- `experiments/training/run_hetero_nll_probe.py`

Research logs:
- `docs/improvement_progress_log_20260519.md`
- `docs/multimarket_framework_branch_manifest_20260525.md`

## Relevant Compact Artifacts

Prefer these report-level directories:
- `data/processed/assets/data_info_vn/history/training_runs/reports/frozen_validation_candidate_20260524/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/academic_ablation_baseline_significance_20260525/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_baseline_significance_20260525/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_ensemble_academic_20260525/`

Keep heavy prediction folders local unless needed for reproducibility:
- `portable_hetero_5seed_preds_20260525_*`
- `hetero_combined_full5_20260521`

## Current Multi-Market Read

| Market | Portable selected rel_score | alpha_rel_score | Read |
| --- | ---: | ---: | --- |
| VN | `0.00613` | `0.00285` | portable positive, but VN-specific anchor is much stronger |
| US | weak/unstable | negative | likely drift-dominated under portable features |
| JP | weak/unstable | slightly negative | no robust stock-selection claim yet |

Interpretation:
- Strong paper claim should remain VN-specific.
- Multi-market section should be framed as a portability stress test.
- The next ambition step is market-specific context adapters, not a single
  one-size-fits-all model.

## Next Implementation Direction

1. Define a common framework interface:
   - portable core feature set,
   - market context adapter,
   - train-only calibration,
   - alpha/demeaned evaluation.
2. Add market adapters:
   - VN: Wyckoff/pressure/breadth context already available.
   - US/JP: add index return, breadth, sector-relative momentum, volatility regime.
3. Keep evaluation frozen:
   - no holdout until the candidate and protocol are fixed,
   - report both raw rel_score and alpha_rel_score,
   - bootstrap paired 21-day fold deltas.

## Cleanup Policy

On this branch:
- remove `.DS_Store` from tracking,
- do not commit raw `.npz` predictions unless explicitly required,
- do not commit every generated run folder,
- promote only compact summary CSV/MD/PNG artifacts that support the paper.
