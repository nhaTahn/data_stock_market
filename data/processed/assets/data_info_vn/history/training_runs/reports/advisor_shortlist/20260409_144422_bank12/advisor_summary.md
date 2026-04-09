# Ngân hàng 12-stock expert

## Candidate shortlist
- Standalone expert: `20260409_144422_bank12_expert` with `lstm_signmag_best_by_val`
- Committee candidate: `overnight_shared_vn100_w20_u64_32_relscore_20260408_173407` + expert
- Committee rule: `method=agree_only`, `weight_expert=0.1`, overlap `ACB,BID,CTG,EIB,HDB,LPB,MBB,STB,TCB,TPB,VCB,VPB`

## Evaluation snapshot
- Standalone `val rel_score`: `0.016085`
- Standalone `test rel_score`: `-0.012054`
- Committee `val rel_score`: `0.016434`
- Committee `test rel_score`: `-0.009919`
- Committee stable-band test median: `-0.012168`
- Committee stable-band test mean: `-0.012260`

## Backtest snapshot
- Standalone best threshold: `0.0`
- Standalone final equity: `1.920791`
- Standalone avg strategy return: `0.000608`
- Committee best threshold: `0.0025`
- Committee final equity: `2.766885`
- Committee avg strategy return: `0.003356`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.5104` vs `actual_pos_rate`: `0.4756`
- Standalone `pred_abs_over_actual_abs`: `0.0847`
- Committee `pred_pos_rate`: `0.5104` vs `actual_pos_rate`: `0.4756`
- Committee `pred_abs_over_actual_abs`: `0.0832`
