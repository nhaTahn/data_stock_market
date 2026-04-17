# F&B committee candidate (sector-base signmag)

## Candidate shortlist
- Standalone expert: `overnight_fnb_w5_mag_base_20260409_100007` with `lstm_best_by_val`
- Market sidecar: `biaspush_signmag_narrow_rawmag_20260409_111710` with `lstm_signmag_best_by_val`
- Committee candidate: `avg(expert, market sidecar)`
- Committee rule: `method=avg`, `weight_expert=0.9`, overlap `KDC,SAB,SBT,VNM`

## Evaluation snapshot
- Standalone `val rel_score`: `0.021013`
- Standalone `test rel_score`: `0.034253`
- Committee `val rel_score`: `0.023671`
- Committee `test rel_score`: `0.051016`
- Committee stable-band test median: `0.050897`
- Committee stable-band test mean: `0.049210`

## Backtest snapshot
- Standalone best threshold: `0.0`
- Standalone final equity: `1.802215`
- Standalone avg strategy return: `0.003725`
- Committee best threshold: `0.0`
- Committee final equity: `1.886135`
- Committee avg strategy return: `0.004301`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.1348` vs `actual_pos_rate`: `0.4158`
- Standalone `pred_abs_over_actual_abs`: `0.1324`
- Committee `pred_pos_rate`: `0.1305` vs `actual_pos_rate`: `0.4163`
- Committee `pred_abs_over_actual_abs`: `0.1225`
