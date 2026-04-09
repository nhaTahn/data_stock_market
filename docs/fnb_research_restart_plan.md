# F&B Research Restart Plan

Mục tiêu của tài liệu này là reset lại hướng nghiên cứu cho cụm `KDC,SAB,SBT,VNM` sau khi các vòng tune gần đây không sửa được bệnh gốc.

## 1. Kết luận đã đủ chắc để giữ

- `objective` vẫn giữ là `rel_score`
- `whole-market LSTM` chưa có bằng chứng đủ tốt để làm đường chính
- `plain LSTM` vẫn là expert mạnh nhất ở F&B nếu đứng một mình
- `signmag` có ích khi làm tín hiệu phụ cho committee
- committee tốt nhất hiện tại là:
  - expert: `overnight_fnb_w5_mag_base_20260409_101741 / lstm_best_by_val`
  - market: `biaspush_signmag_sector_base_20260409_111710 / lstm_signmag_best_by_val`
  - method: `avg`
  - `weight_expert = 0.9`
  - `test rel_score = 0.0510`

## 2. Vấn đề gốc đã thấy rõ

Vấn đề chính hiện tại không phải overfit cổ điển.

- `train rel_score` của nhiều model thấp hơn `val/test`
- `prediction` bị co biên độ rất mạnh so với `actual`
- `prediction > 0` thấp hơn `actual > 0` khá nhiều
- selection instability có tồn tại, nhưng không còn là vấn đề số 1 sau khi thêm diagnostics

Nói ngắn:

- `plain LSTM` = expert tốt hơn nhưng vẫn underfit biên độ
- `signmag` = cân chiều tốt hơn, nhưng còn bảo thủ hơn về biên độ nếu đứng một mình

## 3. Những gì dừng lại

Các hướng sau tạm dừng, không nên tiếp tục đổ compute:

- `rel_score_sharp`
- `rel_score_weighted`
- `residual expert`
- `shared whole-market` như đường chính
- tune lan rộng nhiều sector cùng lúc
- tiếp tục dùng plot mean-aggregate như hình chính để đọc chất lượng model

## 4. Những gì giữ lại

### Main path

- `plain F&B mini-group`
- `window_size=5`
- objective `rel_score`
- selection không chỉ nhìn `best val`, mà phải nhìn thêm bias và split gaps

### Auxiliary path

- `signmag` với `sector features`
- chỉ dùng như competitor hoặc committee signal
- không dùng như main standalone path nếu không vượt plain

### Decision layer

- committee chỉ được giữ nếu:
  - `code_count >= 4`
  - stable weight band không quá hẹp
  - test không chỉ thắng point-wise mà còn cải thiện backtest

## 5. Research questions cho vòng kế tiếp

### Q1. Plain LSTM đang thiếu gì?

Trọng tâm cần trả lời:

- thiếu breadth feature?
- thiếu capacity?
- hay selection đang chọn sai seed do val quá bảo thủ?

### Q2. Signmag giúp ở đâu?

Trọng tâm cần trả lời:

- signmag có thật sự giúp chiều dự báo?
- khi thêm sector features, signmag có giúp committee ổn định hơn không?
- mức cải thiện của signmag có đủ để làm gating model thôi hay có thể thành model chính?

### Q3. Committee đang tốt hơn nhờ gì?

Không chỉ nhìn `rel_score`. Phải tách:

- gain do overlap code tốt
- gain do average/agree rule
- gain do giảm false positives
- hay gain do weight đang quá nghiêng về expert nên market chỉ đóng vai trò nhẹ

## 6. Tiêu chí chọn winner mới

Không dùng một tiêu chí duy nhất.

Thứ tự ưu tiên:

1. `test rel_score`
2. `val rel_score`
3. `pred_abs_over_actual_abs`
4. `pred_pos_rate` lệch bao xa so với `actual_pos_rate`
5. gap `train -> val -> test`
6. backtest equity và directional accuracy

## 7. Batch kế tiếp nên hẹp thế nào

Chỉ nên chạy quanh đúng 3 nhánh:

1. `plain narrow` nhưng nới selection
- không chỉ lấy `best_by_val`
- thử `top2_by_val`, `stable seed band`, hoặc `committee nội bộ plain seeds`

2. `plain + sector features`
- vì sector features đã giúp signmag, cần kiểm tra xem plain có hưởng lợi tương tự không

3. `plain + signmag sector-base committee`
- giữ đây như baseline decision layer để đánh giá xem main-path mới có thực sự thắng baseline committee hiện tại hay không

## 8. Output bắt buộc cho mọi vòng kế tiếp

Mỗi run/candidate phải có:

- `underfit_selection_summary.json`
- `underfit_selection_bias.csv`
- `underfit_selection_gaps.csv`
- backtest summary
- plot scatter và histogram biên độ

Nếu một run không sinh đủ các artifact này thì không coi là candidate chính thức.
