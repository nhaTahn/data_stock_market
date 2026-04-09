# F&B Phase 2: Sector Selection

Phase 2 chỉ bám vào một giả thuyết duy nhất:

- `plain + sector features` là nhánh đúng
- vấn đề hiện tại nằm ở `selection`

## 1. Tại sao đi tiếp theo nhánh này

Từ `phase1_plain_sector_base_20260409_125108`:

- `lstm_best_by_val` fail trên test
- nhưng nhiều seed khác trong cùng run lại dương khá tốt
- run này cũng là run plain có amplitude ratio tốt nhất trong phase 1

Nghĩa là:

- breadth feature đang đi đúng hướng
- nhưng `best_by_val` một seed là quá sắc và chọn sai

## 2. Phase 2 sẽ làm gì

1. Rerun `plain_sector_base` với nhiều seed hơn
2. Scan tất cả cặp seed của family `lstm_seed_*`
3. Chọn:
- `pair_best_by_val`
- `pair_best_by_score`

Trong đó `pair_best_by_score` dùng score theo validation:

- `val_rel_score`
- `val amplitude ratio`
- `val pos_rate_gap`

4. Backtest hai pair-model đó
5. Ghép pair-model tốt nhất với `signmag sector-base`

## 3. Kết quả cần đọc

- `phase2_train_summary.csv`
- `pair_scan_grid.csv`
- `pair_scan_summary.json`
- `phase2_pair_backtest_summary.json`
- `best_committee_summary.json` của committee phase 2

## 4. Tiêu chí để phase 2 được coi là thành công

1. `pair_best_by_score` phải vượt standalone baseline mới của phase 1
2. Nếu ghép với `signmag sector-base`, committee mới phải vượt:
- standalone cũ `0.0343`
- và tốt nhất là áp sát hoặc vượt committee cũ `0.0510`
