# Multi-Market Portable Batch

- Batch: `multimarket_portable_fullcorebal_20260428_r01`
- Data path: `/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_global/history/multimarket_vn_jp_us_portable_fullcorebal_20260428_r01.csv`
- Panel codes: `208`
- Rows: `741805`
- Run summary CSV: `run_summary.csv`
- Per-market summary CSV: `per_market_summary.csv`

## Markets
- `VN`: codes `93`, rows `277096`, window `2012-03-19` -> `2026-03-31`
- `JP`: codes `26`, rows `103619`, window `2010-01-04` -> `2026-03-30`
- `US`: codes `89`, rows `361090`, window `2010-01-04` -> `2026-03-30`

## Validation
- `multimarket_portable_core_marketbalanced`: overall best `lstm_top2_by_val` at rel_score `+0.00350`, dir acc `49.58%`, trained panels `207`

## Validation By Market
- `VN`: best `lstm_signmag_ensemble` from `multimarket_portable_core_marketbalanced` at rel_score `+0.00335`, dir acc `47.93%`, panels `93`
- `JP`: best `arima` from `multimarket_portable_core_marketbalanced` at rel_score `+0.00006`, dir acc `50.92%`, panels `26`
- `US`: best `lstm_signmag_top2_by_val` from `multimarket_portable_core_marketbalanced` at rel_score `+0.00336`, dir acc `52.30%`, panels `89`
