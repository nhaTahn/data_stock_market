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
python3 src/training/scripts/run_fischer_krauss_lstm_tf.py --markets VN,US --model-type attention
```

Key characteristics:
- Input sequences are 240 trading days of standardized daily returns.
- Labels are binary and use the next-day cross-sectional median return as the threshold.
- Standardization is fit on the training split only to avoid look-ahead bias.
- `--model-type baseline` reproduces the plain LSTM classifier, while `--model-type attention` adds temporal attention.
- Outputs such as fit history, validation predictions, and long-short backtest returns are saved under `artifacts/fk_lstm_classifier/`.

## Generated Outputs

- New local experiment outputs should go under `artifacts/`.
- Curated or historically shared outputs live under `data/assets/data_info_vn/history/`.
- This keeps the repo root cleaner and makes `git status` easier to scan.

## Automated Crawling

This repository uses GitHub Actions to run the `run_fetch.py` script daily at `17:00 UTC`. The updated CSV files are then automatically committed and pushed to the repository.
