# Multi-Market Portable Batch

- Batch: `multimarket_portable_20260504_signalfocus_r01`
- Data path: `/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_global/history/multimarket_vn_jp_us_portable_20260504_signalfocus_r01.csv`
- Panel codes: `208`
- Rows: `741805`
- Run summary CSV: `run_summary.csv`
- Per-market summary CSV: `per_market_summary.csv`

## Markets
- `VN`: codes `93`, rows `277096`, window `2012-03-19` -> `2026-03-31`
- `JP`: codes `26`, rows `103619`, window `2010-01-04` -> `2026-03-30`
- `US`: codes `89`, rows `361090`, window `2010-01-04` -> `2026-03-30`

## Validation
- `multimarket_portable_marketplus_signal_focus`: overall best `lstm_signmag_seed_62` at rel_score `+0.00294`, dir acc `49.45%`, trained panels `207`

## Validation By Market
- `VN`: best `lstm_best_by_val` from `multimarket_portable_marketplus_signal_focus` at rel_score `+0.00637`, dir acc `47.17%`, panels `93`
- `JP`: best `lstm_signmag_best_by_val` from `multimarket_portable_marketplus_signal_focus` at rel_score `+0.00130`, dir acc `51.00%`, panels `26`
- `US`: best `lstm_seed_62` from `multimarket_portable_marketplus_signal_focus` at rel_score `+0.00322`, dir acc `50.60%`, panels `89`
