# Best Portable Multi-Market Model

Source run: `/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_global/history/training_runs/multimarket_portable_marketplus_20260504_r01`
Model: `lstm_signmag_top2_by_val`
Top2 components: `lstm_signmag_seed_62, lstm_signmag_seed_52`
Deployable single checkpoint: `lstm_signmag_seed_62`

## Validation

- rel_score: `+0.003549`
- directional_accuracy: `49.82%`
- rmse: `0.023708`
- panels: `208`

## Validation By Market

- `JP`: rel_score `-0.001783`, direction `51.15%`, panels `26`
- `US`: rel_score `+0.002848`, direction `52.00%`, panels `89`
- `VN`: rel_score `+0.004836`, direction `47.32%`, panels `93`

## Why This Is Gold

- Best overall validation rel_score among current portable VN/JP/US runs.
- Keeps all three markets in train/validation and uses no stock identity.
- Signal-focus and compact variants are useful diagnostics, but neither beats this model overall.
- Router selected on train-only did not beat the single portable baseline on validation.

## Caveat

- `lstm_signmag_top2_by_val` is a prediction ensemble, not one checkpoint. The package includes both seed checkpoints used by top2 and the best single checkpoint for deployment fallback.
