# Docs Map

This folder is easier to read if you treat it as three lanes instead of one long archive.

## 1. Current Direction

Open these first if the goal is to improve the model right now:

1. [`current_best_path.md`](current_best_path.md)
2. [`current_research_status.md`](current_research_status.md)
3. [`downtrend_expert_findings.md`](downtrend_expert_findings.md)

These three files define the active benchmark, the rejected paths, and the next justified step.

## 2. Code Reading

Open these first if the goal is to understand where to change code:

1. [`models_code_map.md`](models_code_map.md)
2. [`../main.py`](../main.py)
3. [`../src/models/training/pipeline.py`](../src/models/training/pipeline.py)
4. [`../src/models/components/losses.py`](../src/models/components/losses.py)
5. [`../src/models/architectures/`](../src/models/architectures/)

Use this path for implementation work. Do not start from legacy shim files.

## 3. Saved Results

Open these first if the goal is to inspect outputs instead of code:

1. [`../data/processed/assets/data_info_vn/history/training_runs/README.md`](../data/processed/assets/data_info_vn/history/training_runs/README.md)
2. `data/.../history/training_runs/reports/*/summary.md`
3. [`../data/processed/assets/data_info_vn/gold/README.md`](../data/processed/assets/data_info_vn/gold/README.md)

The repo should prefer compact report summaries over opening raw run folders.

## Active vs Reference

Read often:

- [`current_best_path.md`](current_best_path.md)
- [`current_research_status.md`](current_research_status.md)
- [`models_code_map.md`](models_code_map.md)
- [`vn_reporting_standard.md`](vn_reporting_standard.md)

Read only when needed for background or older experiments:

- [`feature_correlation_report.md`](feature_correlation_report.md)
- [`feature_formula_report.md`](feature_formula_report.md)
- [`vn30_paper_phase1_20260412.md`](vn30_paper_phase1_20260412.md)
- [`vn30_signal_phase2_20260417.md`](vn30_signal_phase2_20260417.md)
- [`pcie_lite_phase1_20260417.md`](pcie_lite_phase1_20260417.md)
- [`lstm_model_glossary.md`](lstm_model_glossary.md)

## Practical Reading Order

If you have only 15 minutes:

1. Read [`current_best_path.md`](current_best_path.md).
2. Read [`models_code_map.md`](models_code_map.md).
3. Open the latest compact report summary under `training_runs/reports/`.

If you are about to edit training code:

1. Read [`current_research_status.md`](current_research_status.md).
2. Open [`../src/models/training/pipeline.py`](../src/models/training/pipeline.py).
3. Open the relevant builder in [`../src/models/architectures/`](../src/models/architectures/).
4. Open the objective in [`../src/models/components/losses.py`](../src/models/components/losses.py).
