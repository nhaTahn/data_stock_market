# F&B phase4 frontier 122+72

## Candidate shortlist
- Standalone expert: `phase3_pair_frontier_20260409_135338` with `lstm_pair_frontier_122_72`
- Committee candidate: `biaspush_signmag_sector_base_20260409_111710` + expert
- Committee rule: `method=agree_only`, `weight_expert=0.9`, overlap `KDC,SAB,SBT,VNM`

## Evaluation snapshot
- Standalone `val rel_score`: `0.018934`
- Standalone `test rel_score`: `0.038834`
- Committee `val rel_score`: `0.020623`
- Committee `test rel_score`: `0.039268`
- Committee stable-band test median: `0.039146`
- Committee stable-band test mean: `0.039009`

## Backtest snapshot
- Standalone best threshold: `0.0`
- Standalone final equity: `2.122354`
- Standalone avg strategy return: `0.002807`
- Committee best threshold: `0.0`
- Committee final equity: `2.122354`
- Committee avg strategy return: `0.002807`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.2489` vs `actual_pos_rate`: `0.4163`
- Standalone `pred_abs_over_actual_abs`: `0.1122`
- Committee `pred_pos_rate`: `0.2489` vs `actual_pos_rate`: `0.4163`
- Committee `pred_abs_over_actual_abs`: `0.1078`
