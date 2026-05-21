`training_runs` stores saved VN experiments plus compact aggregate reports.

## Read This Folder In Two Layers

Do not start by opening random run directories.

Start here:

1. [`../../../../../../docs/current_best_path.md`](../../../../../../docs/current_best_path.md)
2. `reports/*/summary.md`
3. only then open a specific run folder if the compact summary says it matters

## Current Shortlist

The active path is centered on these items:

- prediction anchor: `broad_signmag_prune_general_sector_full_20260424_r04`
- trade challenger: `broad_signmag_prune_phase_ic_sector19_20260425_r09`
- prediction-safe ensemble: `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01`
- rank/router benchmark: `reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01`
- trade-side follow-up: `reports/rank_router_train_selected/anchor_sector19_rank_router_20260427_r01`

For the reasoning behind this shortlist, use:

- [`../../../../../../docs/current_best_path.md`](../../../../../../docs/current_best_path.md)
- [`../../../../../../docs/current_research_status.md`](../../../../../../docs/current_research_status.md)

## What To Open First

Preferred order:

1. `reports/router_weight_grid/*/summary.md`
2. `reports/cross_sectional_ic/*/summary.md`
3. `reports/rank_objective_offline/*/summary.md`
4. `reports/rank_router_train_selected/*/summary.md`
5. `reports/regime_analysis/*/summary.md`

These folders are meant to replace browsing many raw run artifacts.

## Folder Roles

- `reports/`
  Compact aggregate summaries that are worth reading first and are the preferred commit targets.
- `_archive/`
  Old or rejected experiment groups kept only for traceability.
- `active/`, `search_runs/`, `representative_runs/`
  Legacy organization buckets. Do not treat them as the main entrypoint.
- direct run folders such as `broad_*`, `mini_*`, `vn30_*`
  Raw experiment outputs. Open only after a compact summary points to them.

## Git Hygiene

Default rule:

- keep in git: summary markdown/json/csv and top-level README files
- keep local only: predictions, histories, plots, diagnostics, raw backtest CSVs, and most per-run dumps

This keeps the repo readable and reduces accidental pushes of generated noise.
