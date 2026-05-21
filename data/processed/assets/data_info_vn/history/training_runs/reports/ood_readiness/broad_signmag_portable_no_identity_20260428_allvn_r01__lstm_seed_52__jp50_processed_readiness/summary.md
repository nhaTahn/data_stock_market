# OOD Readiness Report

- Run: `broad_signmag_portable_no_identity_20260428_allvn_r01`
- Model: `lstm_seed_52`
- Market: `JP`
- Universe list: `market_lists/jp50.txt`
- Requested codes: `50`
- Clean accepted codes: `26`
- Unique sectors in clean OOD frame: `1`
- Unknown-sector share: `100.0%`
- Test rows: `21366`
- Test codes: `26`
- Stock identity enabled: `False`
- Stock-identity code share known to VN model: `0.0%`
- Stock-identity row share known to VN model: `0.0%`

## Prediction Metrics
- rel_score: `-0.00268`
- directional_accuracy: `49.3%`
- error q2/q8 (E = prediction - actual): `-0.01362` / `+0.01192`
- error mean/std: `-0.00069` / `0.02069`

## Ranking Metrics
- mean daily Spearman IC: `+0.01513`
- IC t-stat: `+1.62`
- positive IC days: `52.7%`
- quartile equity: `1.316`
- quartile hit rate: `51.0%`
- quartile max drawdown: `-21.8%`

## Neutral Fills
- Columns filled with neutral defaults: `sector_momentum_rank_pct`
