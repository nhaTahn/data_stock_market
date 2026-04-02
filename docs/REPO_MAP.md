# Repo Map

This repository mixes three concerns:

1. `data/`: raw and curated market data.
2. `src/`: pipelines and model training code.
3. local experiment outputs: saved under `artifacts/` or selected history folders.

Use the guide below to avoid digging through everything.

## Top Level

```text
data_stock_market/
├── configs/fk_lstm/       Full FK experiment configs for VN/US runs
├── configs/fk_lstm_fast/  Faster FK configs for Colab and early comparison
├── data/                  Raw market CSVs and curated VN history assets
├── market_lists/          Universe definitions used by the fetcher
├── notebook/              Ad-hoc exploration
├── src/data_pipeline/     Fetching and VN dataset preparation
├── src/training/          Model code and runnable training scripts
├── artifacts/             Local model outputs and backtest results (gitignored)
├── run_fetch.py           Main data collection entrypoint
└── README.md              Setup and common commands
```

## Where To Start

- If you want fresh market data: open `run_fetch.py` then `src/data_pipeline/fetch_data.py`.
- If you want curated VN training inputs: open `src/data_pipeline/build_vn_quality_dataset.py`.
- If you want the original TensorFlow LSTM regression flow: open `src/training/tf_lstm/`.
- If you want the Fischer-Krauss style return classifier: open `src/training/fk_lstm_classifier/`.
- If you want benchmark or sweep scripts: open `src/training/README.md`.

## Data Layout

- `data/VN/`, `data/US/`, `data/JP/`, `data/HK/`, `data/KR/`: per-ticker daily CSV files.
- `data/assets/data_info_vn/`: curated VN metadata and historical reports.
- `data/assets/data_info_vn/history/training_runs/`: versioned historical experiment snapshots plus some local generated runs.

## Training Layout

- `src/training/tf_lstm/`: reusable modules for the existing TensorFlow regression pipeline.
- `src/training/fk_lstm_classifier/`: modular binary classification pipeline with optional temporal attention.
- `src/training/fk_lstm_classifier/market_rules.py`: market-specific runtime assumptions such as VN `T+2.5` approximation and long-only evaluation.
- `src/training/scripts/run_vn_fischer_krauss_pack.py`: VN-only walk-forward pack runner for baseline vs attention.
- `src/training/model_benchmark/`: classical and transformer benchmark helpers.
- `src/training/scripts/`: script entrypoints for experiments, ablations, sweeps, and benchmarks.
- `configs/fk_lstm/`: walk-forward experiment presets for baseline vs attention across VN and US.
- `configs/fk_lstm_fast/`: lighter walk-forward presets intended for Colab and faster iteration.
- `notebook/fk_lstm_colab_batch_runner.ipynb`: Colab-friendly batch notebook for FK runs.
- `notebook/fk_lstm_vn_walkforward_colab.ipynb`: Colab-friendly VN-only walk-forward notebook.

## Generated Output Convention

- Use `artifacts/` for new local runs you do not want to commit by default.
- Treat `data/assets/data_info_vn/history/` as curated history, not a scratch directory.
- If a run becomes important enough to keep in git, move or copy only the final promoted outputs.
