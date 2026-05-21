# Multi-Market Portable Batch

- Batch: `multimarket_portable_20260504_compactfix_r01`
- Data path: `/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_global/history/multimarket_vn_jp_us_portable_20260504_compactfix_r01.csv`
- Panel codes: `208`
- Rows: `741805`
- Run summary CSV: `run_summary.csv`
- Per-market summary CSV: `per_market_summary.csv`

## Markets
- `VN`: codes `93`, rows `277096`, window `2012-03-19` -> `2026-03-31`
- `JP`: codes `26`, rows `103619`, window `2010-01-04` -> `2026-03-30`
- `US`: codes `89`, rows `361090`, window `2010-01-04` -> `2026-03-30`

## Validation
- `multimarket_portable_compact_signal`: overall best `lstm_signmag_top2_by_val` at rel_score `+0.00335`, dir acc `49.99%`, trained panels `207`

## Validation By Market
- `VN`: best `lstm_seed_52` from `multimarket_portable_compact_signal` at rel_score `+0.00645`, dir acc `48.19%`, panels `93`
- `JP`: best `lstm_signmag_ensemble` from `multimarket_portable_compact_signal` at rel_score `+0.00145`, dir acc `50.75%`, panels `26`
- `US`: best `lstm_top2_by_val` from `multimarket_portable_compact_signal` at rel_score `+0.00294`, dir acc `51.44%`, panels `89`
