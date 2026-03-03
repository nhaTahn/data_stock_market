# Data Stock Market

This repository contains daily historical stock market data for various markets (VN, US, JP) automatically crawled and updated via GitHub Actions.

## Repository Structure

- `data/` : Contains the historical stock market data, separated by market (`VN/`, `US/`, `JP/`). The data is stored in CSV format.
- `market_lists/` : Contains text files with lists of stock tickers (e.g., VN30, US100) used by the data crawler.
- `src/data_pipeline/fetch_data.py` : The core data pipeline script for downloading and formatting data from `vnstock` and `yfinance`.
- `run_fetch.py` : The entrypoint script to run the data collection.

## Data Schema

The CSV data files in the `data/` directory have the following structure:
- `Date`: The trading date (YYYY-MM-DD).
- `code`: The stock ticker symbol.
- `open`, `high`, `low`, `close`: The OHLVC daily prices.
- `adjust`: Adjusted close price.
- `volume_match`: The matched trading volume.
- `value_match`: The value matched (close * volume).

*Note: For international markets, the schema is formatted to match the VN format for consistency.*

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

## Automated Crawling

This repository uses GitHub Actions to run the `run_fetch.py` script daily at `17:00 UTC`. The updated CSV files are then automatically committed and pushed to the repository.
