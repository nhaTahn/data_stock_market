# Walkthrough + Phase 1 Restart

Tài liệu này ghép hai lớp thông tin:

- các kết quả đã chắc từ vòng nghiên cứu `2026-04-08 -> 2026-04-09`
- kế hoạch `phase 1` hiện tại để tiếp tục tối ưu `rel_score` mà không đổi objective

## 1. Những gì đã được xác nhận

### Feature pruning

- tương quan feature đã được đo lại trên dữ liệu thực
- default feature set đã được giảm từ `28 -> 24`
- các feature bị loại chủ yếu là các cặp trùng thông tin mạnh như:
  - `atr_gap ~ volatility_20`
  - `volume_zscore_20 ~ volume_ratio_20`
  - `ma_20_gap ~ rsi_14/bb_position`
  - `bb_position ~ close_position`

### Data audit

Các lỗi dữ liệu đã sửa và không còn nên bị nghi ngờ là nguyên nhân chính:

- circular leakage trong `alpha_sector`
- look-ahead trong event buffer
- wyckoff epsilon blow-up
- volume ratio chia cho 0

Điều này quan trọng vì các run mạnh sau patch vẫn giữ được tín hiệu, nên kết quả hiện tại đáng tin hơn về khoa học.

### Sector evidence

- `Bất động sản` đã từng vượt `0.03`
- `Ngân hàng` có tín hiệu nhưng chưa đủ mạnh
- `Thực phẩm & đồ uống` là nơi committee hiện giúp tốt nhất
- `whole-market shared LSTM` hiện chưa đủ bằng chứng để làm đường chính

## 2. Trạng thái hiện tại của F&B

### Standalone expert mạnh nhất

- run: `overnight_fnb_w5_mag_base_20260409_101741`
- model: `lstm_best_by_val`
- `test rel_score = 0.0343`

### Auxiliary signmag mạnh nhất

- run: `biaspush_signmag_sector_base_20260409_111710`
- model: `lstm_signmag_best_by_val`
- `test rel_score = 0.0276`

### Decision layer mạnh nhất

- committee: `plain expert + signmag sector-base`
- `test rel_score = 0.0510`

## 3. Vấn đề gốc của F&B

Kết luận hiện tại là:

- chưa thấy overfit cổ điển
- lỗi chính là `amplitude underfit`
- `plain` vẫn là model mạnh nhất nếu đứng một mình
- `signmag` giúp chiều và committee, nhưng chưa đủ để thay plain làm main path

Biểu hiện:

- `pred_abs_over_actual_abs` còn thấp
- `pred_pos_rate` lệch khá xa so với `actual_pos_rate`
- `train rel_score` của nhiều model thấp hơn `val/test`

## 4. Phase 1 nên làm gì

Không đổi objective. Vẫn giữ:

- `loss = rel_score`
- cụm `KDC,SAB,SBT,VNM`

Phase 1 chỉ chạy quanh 3 nhánh:

1. `plain narrow`
- tái lập baseline hẹp với nhiều seed hơn để nới selection

2. `plain narrow higher-capacity`
- kiểm tra xem capacity có phải bottleneck không

3. `plain + sector features`
- kiểm tra xem breadth feature có giúp plain như đã giúp signmag hay không

Sau đó:

- chấm lại `lstm_best_by_val`, `lstm_top2_by_val`, `lstm_ensemble`
- ghép từng run với `signmag sector-base` để xem committee có cải thiện hơn baseline hiện tại không

## 5. Tiêu chí chọn winner trong phase 1

Thứ tự ưu tiên:

1. `committee test rel_score`
2. `standalone test rel_score`
3. `val rel_score`
4. `pred_abs_over_actual_abs`
5. `pred_pos_rate` lệch bao xa so với `actual_pos_rate`
6. gap `train -> val -> test`

Không coi là winner nếu:

- chỉ đẹp ở `val`
- amplitude vẫn co quá mạnh
- committee weight band quá nhọn

## 6. Artifact cần đọc sau phase 1

- `phase1_train_summary.csv`
- `phase1_committee_summary.csv`
- `phase1_candidate_ranking.csv`
- `underfit_selection_summary.json` của từng run
- `threshold_backtest_summary_phase1.json` của từng run
