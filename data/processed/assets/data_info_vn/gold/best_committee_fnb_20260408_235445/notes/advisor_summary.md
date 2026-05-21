# F&B Best Candidate Report

## Candidate shortlist
- Standalone expert: `overnight_fnb_w5_mag_base_20260409_101741` with `lstm_best_by_val` on `KDC,SAB,SBT,VNM`
- Committee candidate: stable-band committee from `VN100 context + F&B expert`
- Committee rule: `method=agree_only`, `weight_expert=0.15`, overlap `KDC,SAB,SBT,VNM`

## Evaluation snapshot
- Standalone `val rel_score`: `0.021013`
- Standalone `test rel_score`: `0.034253`
- Committee `val rel_score`: `0.023285`
- Committee `test rel_score`: `0.049321`
- Committee stable-band test median: `0.051640`
- Committee stable-band test mean: `0.051234`

## Backtest snapshot
- Standalone best threshold: `0.0`
- Standalone final equity: `1.802215`
- Standalone avg strategy return: `0.003725`
- Standalone directional accuracy: `0.695122`
- Committee best threshold: `0.0`
- Committee final equity: `1.802215`
- Committee avg strategy return: `0.003998`
- Committee directional accuracy: `0.703947`

## Bias diagnostics
- Standalone `pred_pos_rate`: `0.1348` vs `actual_pos_rate`: `0.4158`
- Standalone `pred_abs_over_actual_abs`: `0.1324`
- Committee `pred_pos_rate`: `0.1305` vs `actual_pos_rate`: `0.4163`
- Committee `pred_abs_over_actual_abs`: `0.1148`

## Reading notes
- Current best direction is still `F&B mini-group + committee`, not whole-market LSTM.
- Committee improves `rel_score` over standalone on the same 4-code overlap, but still remains far below the target `0.1`.
- The main failure mode is underreaction: prediction amplitude is still much smaller than actual return amplitude.
- This package is suitable for advisor review because it isolates the strongest path and removes the noise from failed whole-market and residual experiments.
