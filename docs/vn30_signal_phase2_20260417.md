# VN30 Signal Phase 2 - 2026-04-17

## Goal

Run a narrow batch to test whether the new `lstm_signal` family adds edge over standard `lstm` on:

1. Existing BDS edge set
2. VN30 bank core
3. Vingroup cluster

The batch holds features fixed at the paper-style representation and changes only:

- `signal_future_steps` in `{1, 3, 5}`
- signal architecture on/off
- optional attention family as a challenger

## Why This Phase Is Narrow

- Phase 1 showed that paper-style representation was worth keeping, but the full denoise bundle was not a clear winner.
- The new `lstm_signal` architecture is now wired into `src/models/training/pipeline.py`.
- The next question is not "search more features", but "does patching + ATL + attention + quantile head improve edge on cleaner sub-universes?"

## Universes

| Universe | Codes | Purpose |
|---|---|---|
| BDS edge | `KOS,DXG,NLG,DIG,TCH,VHM` | Re-test architecture on a group with prior positive sector edge |
| Bank core | `ACB,BID,CTG,MBB,TCB,VCB` | Test sector regime structure inside VN30 |
| VIN cluster | `VIC,VHM,VRE` | Test shared path dependence and co-movement |

## Entry Point

- Script: `python -m experiments.training.run_signal_paper_phase2`
- Output summary: `data/processed/assets/data_info_vn/history/training_runs/overnight_logs/<stamp>_signal_phase2/signal_phase2_summary.json`

## Readout Focus

For each case, compare:

- `lstm_best_test_rel_score`
- `lstm_attention_best_test_rel_score`
- `lstm_signal_best_test_rel_score`
- `signal_gain_vs_lstm_best_test`

Interpretation rule:

- If `future_steps=1` wins but `3/5` do not, patching/dual-preprocess may help while direct multi-step is still too hard.
- If the signal family only helps on BDS and not on VN30 subsets, the architecture is likely learning sector structure rather than general cross-sectional market state.
- If no subset beats standard LSTM, the paper idea is not yet the bottleneck and research should return to signal quality, regime gates, or target design.
