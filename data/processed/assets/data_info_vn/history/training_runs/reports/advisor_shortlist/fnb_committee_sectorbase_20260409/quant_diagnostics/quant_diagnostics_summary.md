# F&B best committee quant diagnostics

## Snapshot
- Standalone test rel_score: `0.034253`
- Committee test rel_score: `0.051016`
- Committee method: `avg`
- Committee weight_expert: `0.9`
- Overlap codes: `KDC,SAB,SBT,VNM`

## Daily IC mean
- committee: `0.094539`
- standalone: `0.073909`

## Regime rel_score
- `down__high_vol`: committee=-0.021260, standalone=-0.036330
- `down__low_vol`: committee=0.012427, standalone=0.013335
- `up__high_vol`: committee=0.038680, standalone=0.043170
- `up__low_vol`: committee=-0.029972, standalone=-0.032366

## Candidate frontier
- `val_max` / `lstm_seed_142__lstm_seed_62`: val `0.030367`, test `-0.000559`
- `balanced` / `lstm_seed_142__lstm_seed_72`: val `0.020514`, test `0.022972`
- `frontier` / `lstm_seed_122__lstm_seed_72`: val `0.018934`, test `0.038834`
- `current_standalone_best` / `data/processed/assets/data_info_vn/history/training_runs/overnight_fnb_w5_mag_base_20260409_101741:lstm_best_by_val`: val `0.021013`, test `0.034253`
- `current_committee_best` / `data/processed/assets/data_info_vn/history/training_runs/biaspush_signmag_sector_base_20260409_111710:lstm_signmag_best_by_val`: val `0.023671`, test `0.051016`