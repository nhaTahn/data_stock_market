# Training Runs Report Cleanup - 2026-04-28

This note records the top-level `training_runs/reports` cleanup applied on `2026-04-28`.

## Goal

Keep the report root focused on the current all-VN decision path while preserving older report bundles in archive form.

## Kept In Place

The following top-level report directories were kept in `training_runs/reports/`:

- `info/`
- `ood_readiness/broad_signmag_portable_no_identity_20260428_allvn_r01__us100_readiness/`
- `portability_ablation/broad_signmag_allvn_20260428_allvn_r01/`

## Archived

Archived to:

- `data/processed/assets/data_info_vn/history/training_runs/_archive/report_cleanup_20260428/`

What was moved:

- older top-level research bundles such as `feature_pruning/`, `regime_analysis/`, `router_*`, `cross_sectional_ic/`, `stock_*`
- older `ood_readiness/` subdirectories not tied to the current all-VN decision
- older `portability_ablation/` subdirectories, including preview and non-all-VN portability batches

## Safety Note

- no run directory under `training_runs/<run_name>/` was removed
- no model checkpoint was deleted
- no current all-VN run artifact was moved out of its run directory

## Result

- `training_runs/reports/` was reduced to a compact current-state view
- older report bundles remain available under `_archive/report_cleanup_20260428/`
