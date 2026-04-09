# F&B RelScore Push Plan

Mục tiêu của batch này là chỉ tập trung vào nhóm `KDC,SAB,SBT,VNM` và cố đẩy `rel_score` vượt vùng hiện tại `0.05`.

## 1. Trạng thái hiện tại

Mốc tham chiếu đang có:

- plain F&B mini-group tốt nhất:
  - [`mini_tpdouong_g06_uncertainty_sidecar`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/mini_tpdouong_g06_uncertainty_sidecar)
  - `lstm_best_by_val`
  - `test rel_score = 0.0343`
- shared committee mạnh nhất:
  - [`confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/committee_experiments/confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb)
  - stable-band test median khoảng `0.0516`

## 2. Bệnh gốc

Plain LSTM đang có:

- bias âm rõ
- biên độ dự báo co mạnh
- tỷ lệ `prediction > 0` thấp hơn nhiều so với `actual > 0`

`signmag` giúp cân bằng chiều hơn nhưng chưa đủ mạnh để thắng plain LSTM nếu đứng một mình.

Vì vậy batch này sẽ tối ưu:

- plain LSTM quanh vùng đang thắng
- signmag như một bias-corrector
- committee nội bộ và committee với shared context như lớp chấm cuối

## 3. Không làm gì trong batch này

- không mở thêm sector khác
- không thêm quantile làm model chính
- không thêm residual expert
- không thêm whole-market search

## 4. Matrix nên chạy

Universe cố định:

- `KDC,SAB,SBT,VNM`

Feature set cố định:

- `volume_ratio_20`
- `close_position`
- `lower_shadow`
- `alpha_sector`
- `vingroup_momentum`
- `vnindex_return`
- `a_d_ratio`
- `day_of_week`
- `rsi_14`

Run set:

1. `w5_mag_base`
- window `5`
- units `48,24`
- lr `0.0002`
- dropout `0.05`
- magnitude sample weight như baseline

2. `w5_mag_tighter_lr`
- như trên
- lr `0.00015`

3. `w5_mag_higher_units`
- window `5`
- units `64,32`
- lr `0.0002`

4. `w5_nomag`
- window `5`
- units `48,24`
- sample weight `none`

5. `w7_mag`
- window `7`
- units `48,24`
- magnitude sample weight

6. `w10_mag`
- window `10`
- units `48,24`
- magnitude sample weight

## 5. Mỗi run sẽ được chấm theo 3 tầng

1. `best_by_val` của chính run
2. committee nội bộ:
- expert: `lstm_best_by_val,lstm_ensemble`
- market: `lstm_signmag_best_by_val,lstm_signmag_top2_by_val,lstm_signmag_ensemble,lstm_quantile_best_by_val`

3. committee với shared context:
- market run cố định: `confirm_vn100_fnb_committee_20260408_235445_r01`
- market models: `lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble`

## 6. Cách chọn winner

Ưu tiên:

1. stable-band committee median
2. committee test rel_score
3. plain run best-by-val test rel_score
4. bias stats

Không coi là winner nếu:

- val quá thấp
- stable band quá hẹp
- code_count hẹp
- bias vẫn lệch nặng

## 7. Kỳ vọng thực tế

Mốc nên đọc:

- `0.06` = tiến bộ rõ
- `0.08` = rất đáng giữ
- `0.10` = thành công rõ

`0.2` hiện chưa phải mục tiêu hợp lý cho một batch F&B duy nhất.
