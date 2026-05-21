# Best VN30 Broad Model (29 stocks)

Source run: `data/processed/assets/data_info_vn/history/training_runs/vn30_paper_repr_20260412_paper_phase1`
Model: `lstm_signmag_seed_52`
Stocks: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE

This package uses the **test split** only.
`E` is defined as `prediction - actual` after the same per-stock temporal alignment used by `rel_score`.

## Scores

- `rel_score`: 0.001336
- `base_score`: 0.028307
- `abs_score`: 0.028269
- `raw_test_rows`: 8829
- `aligned_error_rows`: 8771

## Error Summary

- `min`: -0.071022
- `q25`: -0.010111
- `median`: -0.000193
- `q75`: 0.009044
- `max`: 0.087637
- `mean`: -0.000849
- `std`: 0.022677

## Note

This is the best retained broad-panel VN30 run. Test rel_score is positive but still small, so it should be presented as a weak broad-market result, not a strong tradable edge.
