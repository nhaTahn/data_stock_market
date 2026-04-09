# Overnight RelScore Push Plan

Mục tiêu của batch này là đẩy `rel_score` vượt vùng hiện tại `0.03 -> 0.05` lên vùng cao hơn. Mốc `0.1` là mục tiêu tấn công. Mốc `0.2` là stretch goal, không nên coi là baseline kỳ vọng cho một đêm chạy.

## 1. Kết luận xuất phát

Từ các run mạnh nhất hiện tại:

- F&B mini-group mạnh nhất vẫn là:
  - [`mini_tpdouong_g06_uncertainty_sidecar`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/mini_tpdouong_g06_uncertainty_sidecar)
  - `lstm_best_by_val`
  - `test rel_score = 0.0343`
- Shared committee mạnh nhất hiện tại là:
  - [`confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb)
  - stable-band `test rel_score` median khoảng `0.0516`
- BĐS có ceiling tốt ở `signmag`, nhưng chưa ổn định:
  - [`mini_bat_ong_san_g01_return_w20_pruned_v2`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/mini_bat_ong_san_g01_return_w20_pruned_v2)
  - `lstm_signmag_seed_62 test rel_score = 0.0341`

## 2. Vấn đề gốc

F&B plain LSTM đang có:

- bias âm rõ trên `val/test`
- biên độ dự báo bị co rất mạnh
- tỷ lệ `prediction > 0` thấp hơn nhiều so với `actual > 0`

BĐS thì ngược lại:

- bias dương nhẹ
- biên độ dự báo lớn hơn
- `signmag` có tiềm năng hơn `plain`

Điều này cho thấy overnight nên đi theo hai nhánh khác nhau:

- F&B: sửa `underreaction / short bias`
- BĐS: tìm `signmag` ổn định hơn bằng seed + window

## 3. Ma trận overnight nên chạy

### F&B

Universe:

- `KDC,SAB,SBT,VNM`

Feature set:

- giữ nguyên feature set của run mạnh nhất hiện tại

Runs:

1. `fnb_w5_plain_more_seeds`
- window `5`
- units `48,24`
- lr `0.0002`
- dropout `0.05`
- seeds `42,52,62,72,82,92`
- sample weight `none`

2. `fnb_w5_plain_magweight`
- như trên
- sample weight `magnitude`

3. `fnb_w10_plain_more_seeds`
- window `10`
- units `48,24`
- lr `0.0002`
- dropout `0.05`
- seeds `42,52,62,72,82,92`

4. `fnb_w10_plain_magweight`
- như trên
- sample weight `magnitude`

### BĐS

Universe:

- `KOS,DXG,NLG,DIG,TCH,VHM`

Feature set:

- giữ nguyên feature set pruned đã thắng hiện tại

Runs:

5. `bds_w20_more_seeds`
- window `20`
- units `64,32`
- lr `0.0003`
- dropout `0.08`
- seeds `42,52,62,72,82,92,99,109`

6. `bds_w15_more_seeds`
- window `15`
- units `64,32`
- lr `0.0003`
- dropout `0.10`
- seeds `42,52,62,72,82,92,99,109`

### Shared context + committee

7. `shared_vn100_w20_fnb_committee_more_seeds`
- universe `VN100`
- window `20`
- units `64,32`
- lr `0.0005`
- dropout `0.05`
- seeds `42,52,62,72,82,92`
- committee preset `fnb`
- selection mode `stable_band`

## 4. Cách đọc kết quả sáng mai

Ưu tiên theo thứ tự:

1. `committee_test_rel_score`
2. `best_by_val test rel_score`
3. bias stats:
   - `test_pred_pos_rate`
   - `test_actual_pos_rate`
   - `test_pred_abs_over_actual_abs`
4. `stable_weight_count` nếu là committee

Tôi không xem một run là thắng chỉ vì `test rel_score` cao nếu:

- `val rel_score` quá thấp
- hoặc `code_count` quá hẹp
- hoặc bias quá lệch so với actual

## 5. Kỳ vọng thực tế

Batch này có xác suất tốt nhất để:

- đẩy F&B từ vùng `0.05` lên cao hơn nếu sample weighting hoặc seed spread sửa được short bias
- đẩy BĐS từ vùng `0.03` lên gần `0.05`
- xác nhận xem shared committee có tăng được lên trên baseline `0.0516` hay không

Kỳ vọng hợp lý:

- vượt `0.06` là tiến bộ tốt
- chạm `0.08` là rất đáng giữ
- `0.1` là thành công rõ
- `0.2` chỉ nên coi là trần tham vọng, không phải mức chuẩn để đánh giá batch này
