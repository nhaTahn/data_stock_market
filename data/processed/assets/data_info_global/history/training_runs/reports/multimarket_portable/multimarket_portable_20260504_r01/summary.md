# Multi-Market Portable Batch

- Batch: `multimarket_portable_20260504_r01`
- Data path: `/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_global/history/multimarket_vn_jp_us_portable_20260504_r01.csv`
- Panel codes: `208`
- Rows: `741805`
- Run summary CSV: `run_summary.csv`
- Per-market summary CSV: `per_market_summary.csv`

## Markets
- `VN`: codes `93`, rows `277096`, window `2012-03-19` -> `2026-03-31`
- `JP`: codes `26`, rows `103619`, window `2010-01-04` -> `2026-03-30`
- `US`: codes `89`, rows `361090`, window `2010-01-04` -> `2026-03-30`

## Validation
- `multimarket_portable_marketplus`: overall best `lstm_signmag_top2_by_val` at rel_score `+0.00355`, dir acc `49.82%`, trained panels `207`

## Validation By Market
- `VN`: best `lstm_signmag_top2_by_val` from `multimarket_portable_marketplus` at rel_score `+0.00484`, dir acc `47.32%`, panels `93`
- `JP`: best `lstm_signmag` from `multimarket_portable_marketplus` at rel_score `+0.00077`, dir acc `51.45%`, panels `26`
- `US`: best `lstm_signmag_best_by_val` from `multimarket_portable_marketplus` at rel_score `+0.00481`, dir acc `51.91%`, panels `89`
