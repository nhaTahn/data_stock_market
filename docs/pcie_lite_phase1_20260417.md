# PCIE-Lite Phase 1 - 2026-04-17

## Goal

Test a lighter patched-channel architecture that keeps:

- dual input idea through explicit level and percentage-change channels
- MA10 and MA20 smoothing on close
- patching
- shared linear channel projection
- shallow 2-layer LSTM

and removes:

- attention
- quantile head
- pinball loss
- deep per-channel MLP mixing

The evaluation metric remains `rel_score`.

## Entry Point

- Script: `scripts/run_pcie_lite_phase1.py`
- Summary output:
  `data/processed/assets/data_info_vn/history/training_runs/overnight_logs/<stamp>_pcie_lite_phase1/pcie_lite_phase1_summary.json`

## Universes

| Universe | Codes | Purpose |
|---|---|---|
| BDS edge | `KOS,DXG,NLG,DIG,TCH,VHM` | Re-test on an existing positive cluster |
| Bank core | `ACB,BID,CTG,MBB,TCB,VCB` | Sector regime structure inside VN30 |
| VIN cluster | `VIC,VHM,VRE` | Shared path dependence and group co-movement |

## Initial Grid

- `pcie_lite_future_steps`: `3`, `5`
- `pcie_lite_patch_length`: `5`, `10`
- `pcie_lite_patch_stride`: `5`

## Readout Focus

For each case compare:

- `lstm_best_test_rel_score`
- `lstm_attention_best_test_rel_score`
- `lstm_pcie_lite_best_test_rel_score`
- `lstm_signmag_best_test_rel_score`
- `pcie_gain_vs_lstm_best_test`

Interpretation:

- If PCIE-lite helps mainly on `VIN`, the value is likely in smoother shared-cluster structure rather than general market universality.
- If `patch_length=10` beats `5`, the model likely benefits from coarser trend extraction.
- If PCIE-lite still loses to plain/signmag on all three universes, then architecture is not the bottleneck and research should return to target design, regime gates, and signal selection.
