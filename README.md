# Data Stock Market

This repository contains daily historical stock market data for various markets (VN, US, JP) automatically crawled and updated via GitHub Actions.

For a fast orientation pass, start with [docs/REPO_MAP.md](/Users/lap15111/Documents/research-paper/data_stock_market/docs/REPO_MAP.md).

## Repository Structure

- `data/`: Historical market CSVs plus curated VN data assets.
- `market_lists/`: Market universes used by the crawler.
- `src/data_pipeline/`: Fetching and VN dataset preparation.
- `src/training/`: Model packages and training scripts. See [src/training/README.md](/Users/lap15111/Documents/research-paper/data_stock_market/src/training/README.md).
- `artifacts/`: Local experiment outputs and backtests. This folder is gitignored.
- `run_fetch.py`: Main entrypoint for data collection.

## Common Entry Points

- Update market data: `python run_fetch.py --market ALL`
- Build the VN quality dataset: `python3 src/data_pipeline/build_vn_quality_dataset.py`
- Run the TensorFlow LSTM classifier: `python3 src/training/scripts/run_fischer_krauss_lstm_tf.py --markets VN --model-type attention`
- Run the config-driven FK batch suite: `python3 src/training/scripts/run_fk_batch.py --config-dir configs/fk_lstm`
- Run the lighter Colab-friendly FK suite: `python3 src/training/scripts/run_fk_batch.py --config-dir configs/fk_lstm_fast`

## Data Schema

The CSV data files in the `data/` directory have the following structure:
- `Date`: The trading date (YYYY-MM-DD).
- `code`: The stock ticker symbol.
- `open`, `high`, `low`, `close`: The OHLVC daily prices.
- `adjust`: Adjusted close price.
- `volume_match`: The matched trading volume.
- `value_match`: Observed matched value from the source when available.
- `value_match_est`: Estimated matched value (`close * volume_match`).
- `value_match_imputed`: `1` when `value_match` is missing/zero despite matched volume, so downstream analysis should rely on `value_match_est` instead.

*Note: For international markets fetched from Yahoo Finance, `value_match` is not observed from the source, so it is left blank and `value_match_est` is the usable proxy.*

## How to Run Locally

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the crawler:
   ```bash
   python run_fetch.py --market ALL
   ```
   You can also specify `--market VN`, `--market US`, or `--market JP`. To only fetch custom tickers from `watchlist.txt`, add the `--watchlist-only` flag.

## LSTM Return Classification

The repository also includes a modular TensorFlow/Keras implementation of a Fischer-Krauss style stock ranking model using daily returns:

```bash
python3 src/training/scripts/run_fischer_krauss_lstm_tf.py --config configs/fk_lstm/vn_attention.json
```

Key characteristics:
- Input sequences are 240 trading days of standardized daily returns.
- Labels are binary and use the forward-horizon cross-sectional median return as the threshold.
- Standardization is fit on the training split only to avoid look-ahead bias.
- `--model-type baseline` reproduces the plain LSTM classifier, while `--model-type attention` adds temporal attention.
- `evaluation_mode=walk_forward` uses train/validation/test folds with transaction-cost-aware long-short evaluation.
- Outputs such as fit history, validation predictions, holdings, fold summaries, and long-short backtest returns are saved under `artifacts/fk_lstm_classifier/`.

Vietnam-specific market rules are now encoded in the FK pipeline:
- VN defaults to `long-only` evaluation because cash-equity short selling is not assumed.
- VN defaults to `forward_horizon_days=2` as a daily-bar approximation of `T+2.5` settlement sellability.
- VN defaults to `sell_tax_bps=10.0` and leaves broker commission configurable via `buy_cost_bps` / `sell_cost_bps`.
- VN research presets now load the curated `balanced` profile and apply basic microstructure filters:
  - `min_daily_value_traded = 1e9`
  - `min_adv20_value_traded = 2e9`
  - `max_position_adv_fraction = 5%`
  - `portfolio_notional = 1e9`
  - block `limit-up-like` entries on daily bars
  - exclude rows with hard issue flags when that metadata exists
- These resolved rules are saved per run in `market_rules.json`.

Preset research configs live under `configs/fk_lstm/`:
- `vn_baseline.json`
- `vn_attention.json`
- `us_baseline.json`
- `us_attention.json`

Faster comparison presets for Colab live under `configs/fk_lstm_fast/`:
- `vn_baseline_fast.json`
- `vn_attention_fast.json`
- `us_baseline_fast.json`
- `us_attention_fast.json`

The batch runner supports resumable Colab-style execution:
- `--output-root /path/to/drive/fk_lstm_classifier`
- `--skip-existing`

For a notebook-based Colab workflow, open [fk_lstm_colab_batch_runner.ipynb](/Users/lap15111/Documents/research-paper/data_stock_market/notebook/fk_lstm_colab_batch_runner.ipynb).
For the VN-only walk-forward pack on Colab, open [fk_lstm_vn_walkforward_colab.ipynb](/Users/lap15111/Documents/research-paper/data_stock_market/notebook/fk_lstm_vn_walkforward_colab.ipynb).

## Generated Outputs

- New local experiment outputs should go under `artifacts/`.
- Curated or historically shared outputs live under `data/assets/data_info_vn/history/`.
- This keeps the repo root cleaner and makes `git status` easier to scan.

## Automated Crawling

This repository uses GitHub Actions to run the `run_fetch.py` script daily at `17:00 UTC`. The updated CSV files are then automatically committed and pushed to the repository.
