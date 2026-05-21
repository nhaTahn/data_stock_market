# Data Stock Market

VN-focused research repo for market-data build, feature engineering, sequence models, evaluation, backtests, and report packaging.

## Start Here

If you want the current research direction:

1. [`docs/current_best_path.md`](docs/current_best_path.md)
2. [`docs/current_research_status.md`](docs/current_research_status.md)
3. [`docs/downtrend_expert_findings.md`](docs/downtrend_expert_findings.md)

If you want the shortest repo map:

1. [`docs/README.md`](docs/README.md)
2. [`docs/models_code_map.md`](docs/models_code_map.md)

If you want saved results instead of code:

1. [`data/processed/assets/data_info_vn/history/training_runs/README.md`](data/processed/assets/data_info_vn/history/training_runs/README.md)
2. `data/.../history/training_runs/reports/*/summary.md`
3. [`data/processed/assets/data_info_vn/gold/README.md`](data/processed/assets/data_info_vn/gold/README.md)

## Repo Layout

- [`configs/`](configs/): runtime defaults such as `lstm_config.json`
- [`docs/`](docs/): active research notes and reading maps
- [`experiments/`](experiments/): batch runners, offline analyses, and packaging scripts
- [`src/`](src/): reusable pipeline, model, evaluation, reporting, and visualization code
- [`data/`](data/): raw data, processed datasets, saved runs, and compact reports
- [`notebooks/`](notebooks/): exploratory inspection only

## Code Entry Points

- [`main.py`](main.py): main CLI entrypoint
- [`src/models/training/pipeline.py`](src/models/training/pipeline.py): end-to-end training orchestration
- [`src/models/components/losses.py`](src/models/components/losses.py): training objectives
- [`src/models/architectures/`](src/models/architectures/): model families
- [`src/evaluation/metric.py`](src/evaluation/metric.py): final `rel_score` evaluator

## Current Scope

The active path is intentionally narrow:

- target mode: `return`
- main evaluation metric: `rel_score`
- trusted standalone anchor: `general_sector_full`
- current improvement direction: cross-sectional ranking and regime-aware routing
- avoid expanding architecture complexity until the simpler rank/router path stalls

## Git Hygiene

The repo should prefer compact summaries over raw run dumps:

- commit: docs, curated config/code changes, compact report summaries
- keep local: raw predictions, histories, plots, diagnostics, and most per-run artifacts

`.gitignore` is biased toward keeping `training_runs/reports/*/summary.*` and similar compact outputs while leaving heavy generated files untracked.
