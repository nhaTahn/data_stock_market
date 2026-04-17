# OOD VN30 Broad Model on VN100

- source_package: `data/processed/assets/data_info_vn/gold/best_vn30_broad_lstm_signmag_seed_52_20260412`
- checkpoint: `model_signmag_seed_52.keras`
- rel_score: `0.001035`
- base_score: `0.030671`
- abs_score: `0.030639`
- requested_universe_count: `100`
- available_test_code_count: `94`
- missing_from_dataset_count: `6`

Models were trained with stock-identity one-hot. Codes outside the original training universe are inferred with all-zero identity, so this package is an OOD stress test, not a clean validation run.
