# OOD Readiness Report

- Run: `broad_signmag_portable_no_identity_20260428_allvn_r01`
- Model: `lstm_signmag_seed_52`
- Market: `US`
- Universe list: `market_lists/us100.txt`
- Requested codes: `100`
- Clean accepted codes: `89`
- Unique sectors in clean OOD frame: `1`
- Unknown-sector share: `100.0%`
- Test rows: `75017`
- Test codes: `89`
- Stock identity enabled: `False`
- Stock-identity code share known to VN model: `0.0%`
- Stock-identity row share known to VN model: `0.0%`

## Prediction Metrics
- rel_score: `-0.00490`
- directional_accuracy: `50.3%`
- error q2/q8 (E = prediction - actual): `-0.01127` / `+0.01025`
- error mean/std: `-0.00041` / `0.01736`

## Ranking Metrics
- mean daily Spearman IC: `+0.00779`
- IC t-stat: `+1.14`
- positive IC days: `51.2%`
- quartile equity: `1.368`
- quartile hit rate: `50.7%`
- quartile max drawdown: `-14.7%`

## Neutral Fills
- Columns filled with neutral defaults: `sector_momentum_rank_pct`
