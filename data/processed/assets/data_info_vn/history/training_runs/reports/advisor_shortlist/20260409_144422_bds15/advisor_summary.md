# BĐS 15-stock expert

## Candidate shortlist
- Standalone expert: `20260409_144422_bds15_expert` with `lstm_signmag_best_by_val`
- Committee candidate: `overnight_shared_vn100_w20_u64_32_relscore_20260408_173407` + expert
- Committee rule: `method=agree_only`, `weight_expert=0.65`, overlap `BCM,DIG,DXG,KBC,KDH,KOS,NLG,PDR,SIP,SZC,TCH,VHM,VIC,VPI,VRE`

## Evaluation snapshot
- Standalone `val rel_score`: `0.012859`
- Standalone `test rel_score`: `0.001411`
- Committee `val rel_score`: `0.015800`
- Committee `test rel_score`: `0.002368`
- Committee stable-band test median: `0.002346`
- Committee stable-band test mean: `0.002058`

## Backtest snapshot
- Standalone best threshold: `0.0025`
- Standalone final equity: `2.922140`
- Standalone avg strategy return: `0.002035`
- Committee best threshold: `0.0`
- Committee final equity: `2.786106`
- Committee avg strategy return: `0.000916`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.4719` vs `actual_pos_rate`: `0.4751`
- Standalone `pred_abs_over_actual_abs`: `0.1104`
- Committee `pred_pos_rate`: `0.4719` vs `actual_pos_rate`: `0.4751`
- Committee `pred_abs_over_actual_abs`: `0.1016`
