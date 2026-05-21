# Current Best Time-Window Histogram Report

Report này gom các package đang được `gold_index.json` xem là `current_best_*`.

Các cửa sổ thời gian:

- `train`: `[..., 2020-03-31]`
- `in_sample`: `[2020-04-01, 2022-11-15]`

Quy ước:

- `E = prediction - actual`
- `rel_score` dùng cùng công thức hiện tại của repo, tính lại riêng trong từng cửa sổ
- histogram `rel_score` bên dưới dùng stabilized local proxy để tránh méo mạnh khi `|actual|` quá nhỏ
- `out_sample` không được đưa vào report mặc định

Summary CSV: `core/current_best_time_window_summary.csv`

## Packages

### `best_committee_fnb_20260408_235445`

- roles: `current_best_vn100_cluster`
- model: `committee_best_by_val`
- package_dir: `best_committee_fnb_20260408_235445`
- plot: `plots/best_committee_fnb_20260408_235445__committee_best_by_val__time_hist.png`

![best_committee_fnb_20260408_235445](plots/best_committee_fnb_20260408_235445__committee_best_by_val__time_hist.png)

### `best_vin_cluster_lstm_attention_20260417`

- roles: `current_best_overall, current_best_vn30_cluster`
- model: `lstm_attention`
- package_dir: `best_vin_cluster_lstm_attention_20260417`
- plot: `plots/best_vin_cluster_lstm_attention_20260417__lstm_attention__time_hist.png`

![best_vin_cluster_lstm_attention_20260417](plots/best_vin_cluster_lstm_attention_20260417__lstm_attention__time_hist.png)

### `best_vn30_broad_lstm_signmag_seed_52_20260412`

- roles: `current_best_vn30_broad`
- model: `lstm_signmag_seed_52`
- package_dir: `best_vn30_broad_lstm_signmag_seed_52_20260412`
- plot: `plots/best_vn30_broad_lstm_signmag_seed_52_20260412__lstm_signmag_seed_52__time_hist.png`

![best_vn30_broad_lstm_signmag_seed_52_20260412](plots/best_vn30_broad_lstm_signmag_seed_52_20260412__lstm_signmag_seed_52__time_hist.png)
