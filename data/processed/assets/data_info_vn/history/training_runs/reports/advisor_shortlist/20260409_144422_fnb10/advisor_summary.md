# F&B 10-stock expert

## Candidate shortlist
- Standalone expert: `20260409_144422_fnb10_expert` with `lstm_ensemble`
- Committee candidate: `overnight_shared_vn100_w20_u64_32_relscore_20260408_173407` + expert
- Committee rule: `method=agree_only`, `weight_expert=0.1`, overlap `ANV,DBC,HAG,KDC,MSN,PAN,SAB,SBT,VHC,VNM`

## Evaluation snapshot
- Standalone `val rel_score`: `0.002925`
- Standalone `test rel_score`: `0.006152`
- Committee `val rel_score`: `0.006165`
- Committee `test rel_score`: `0.005661`
- Committee stable-band test median: `0.006303`
- Committee stable-band test mean: `0.006378`

## Backtest snapshot
- Standalone best threshold: `0.0025`
- Standalone final equity: `3.185318`
- Standalone avg strategy return: `0.005830`
- Committee best threshold: `0.0025`
- Committee final equity: `2.862980`
- Committee avg strategy return: `0.006597`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.4605` vs `actual_pos_rate`: `0.4478`
- Standalone `pred_abs_over_actual_abs`: `0.0781`
- Committee `pred_pos_rate`: `0.4605` vs `actual_pos_rate`: `0.4478`
- Committee `pred_abs_over_actual_abs`: `0.0755`
