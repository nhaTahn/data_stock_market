# OOD VIN Cluster Model on VN100

- source_package: `data/processed/assets/data_info_vn/gold/best_vin_cluster_lstm_attention_20260417`
- checkpoint: `model_attention.keras`
- rel_score: `-0.010964`
- base_score: `0.030364`
- abs_score: `0.030697`
- requested_universe_count: `100`
- available_test_code_count: `94`
- missing_from_dataset_count: `6`

Models were trained with stock-identity one-hot. Codes outside the original training universe are inferred with all-zero identity, so this package is an OOD stress test, not a clean validation run.
