# Whole-Market LSTM Plan

Tài liệu này trả lời hai câu hỏi:

1. Sau overnight tới nên để ý sector nào?
2. Làm sao kiểm tra xem LSTM có phù hợp để dự báo chung cho cả thị trường hay không?

## 1. Sector watchlist nên theo dõi

Các sector nên đưa vào watchlist tiếp theo trong repo hiện tại là:

- `Ngân hàng`
- `Dịch vụ tài chính`
- `Xây dựng và Vật liệu`
- `Điện, nước & xăng dầu khí đốt`

Tên sector này khớp đúng với cột `sector` trong:

- [`vn_gold_recommended.csv`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/vn_gold_recommended.csv)

Lý do chọn:

- `Ngân hàng`: overlap với `VN30/VN100` tốt, thích hợp để kiểm tra market-context thật sự có giúp hay không.
- `Dịch vụ tài chính`: là proxy gần nhất cho nhóm chứng khoán hiện tại; nếu fail liên tục thì committee kiểu “dẫn sóng tài chính” chưa đáng tin.
- `Xây dựng và Vật liệu`: có tính chu kỳ và thường đi cùng các pha risk-on/risk-off, phù hợp để test regime sensitivity.
- `Điện, nước & xăng dầu khí đốt`: là nhóm phòng thủ/tiện ích, có thể giúp kiểm tra xem shared LSTM chỉ học momentum hay có học được state khác nhau của thị trường.

## 2. Bài học hiện tại trước khi mở rộng

Hiện tại:

- `F&B` là case tốt nhất cho mini-group/committee
- `BĐS` có ceiling tốt nhưng chưa ổn định
- `Ngân hàng` có overlap đẹp nhưng committee test còn thấp
- `Dịch vụ tài chính` đang yếu

Điều này cho thấy:

- shared market model hiện chưa đủ mạnh để coi là model chung cho toàn thị trường
- nhưng committee với context rộng vẫn có ích ở một vài cụm
- vì vậy hướng đúng không phải “train giant LSTM ngay”, mà là `đo suitability theo các phase`

## 3. Mục tiêu của nhánh whole-market

Không phải chỉ hỏi:

- “shared model test rel_score có cao không?”

Mà phải hỏi:

- shared model có tạo context hữu ích cho nhiều sector không?
- shared model có ổn định hơn mini-group ở nhiều regime không?
- shared model có giảm được số case đi ngược thị trường không?

## 4. Success criteria để nói LSTM hợp cho cả thị trường

Tôi chỉ coi `shared-market LSTM` là đáng theo tiếp nếu đạt cùng lúc:

1. `shared standalone` không âm và tốt hơn mức noise:
- target tối thiểu: `test rel_score > 0.02`
- target tốt hơn: `> 0.03`

2. `sector activation` không quá hẹp:
- ít nhất `3` sector trong watchlist có `test rel_score > 0`
- hoặc `2` sector vượt `0.03`

3. `committee usefulness` phải có breadth:
- `code_count >= 3`
- không dựa vào đúng `1` mã trụ như `VHM`

4. `bias stats` không quá lệch:
- `test_pred_pos_rate` không lệch xa `test_actual_pos_rate`
- `test_pred_abs_over_actual_abs` không quá nhỏ như `0.10-0.15`

Nếu các điều kiện này không đạt, thì kết luận hợp lý là:

- LSTM hiện phù hợp hơn cho `sector/mini-group experts`
- chưa phù hợp để làm `1 model chung cho cả thị trường`

## 5. Phase plan

### Phase A: Sector probes

Mục tiêu:

- đo xem từng sector có phản ứng khác nhau với cùng một architecture không

Nên chạy:

- `Ngân hàng`
- `Dịch vụ tài chính`
- `Xây dựng và Vật liệu`
- `Điện, nước & xăng dầu khí đốt`

Cho mỗi sector:

- 1 run `sector-wide`
- 1 hoặc 2 run `mini-group` nếu có overlap tốt trong `VN100`
- 1 committee với shared context nếu overlap `>= 3`

### Phase B: Shared-market probes

Mục tiêu:

- đo xem shared model có học được market state chứ không chỉ average noise

Nên chạy:

1. `shared_vn30`
2. `shared_vn100`

Mỗi shared run nên được đọc theo:

- standalone `rel_score`
- committee gain cho từng sector
- bias stats

### Phase C: Whole-market decision

Mục tiêu:

- chốt xem nên đi theo:
  - `shared-market + sector experts`
  - hay tiếp tục `sector-first / mini-group-first`

Quy tắc ra quyết định:

- nếu shared standalone yếu nhưng committee giúp nhiều sector:
  - giữ hướng `market context + experts`
- nếu shared standalone yếu và committee chỉ giúp `1` sector:
  - quay về `sector-first`
- nếu shared standalone đạt dương rộng và committee ổn ở nhiều sector:
  - bắt đầu xem LSTM là ứng viên cho “whole-market backbone”

## 6. Overnight matrix nên ưu tiên sau batch hiện tại

Sau batch relscore push hiện tại, nên mở overnight tiếp theo theo ma trận này:

### Shared runs

- `shared_vn100_w20_relscore_more_seeds`
- `shared_vn100_w60_relscore_more_seeds`

Lý do:

- `w20` đang là candidate tốt nhất
- `w60` giúp kiểm tra xem sector chậm như tiện ích và vật liệu có cần ngữ cảnh dài hơn không

### Sector committees

Presets nên có:

- `bank`
- `chung` hoặc bản thay thế tốt hơn cho nhóm chứng
- `xaydung_vatlieu`
- `dien_nuoc_gas`

Ghi chú:

- hai preset cuối hiện chưa có runner riêng trong repo
- nên thêm sau khi xác định được mini-group/sector baseline sạch cho từng nhóm

## 7. Cách đọc kết quả sáng hôm sau

Thứ tự đọc:

1. `shared standalone`
2. `committee_rotation_active.csv`
3. `committee_rotation_all.csv`
4. `predictions.csv` để xem bias stats nếu có run đáng chú ý

Các câu hỏi cần trả lời:

- shared model nào có `val/test` cùng dấu?
- sector nào gain so với expert?
- gain đó có đến từ `>= 3` mã hay chỉ `1-2` mã?
- sector phòng thủ như `Điện, nước & xăng dầu khí đốt` có phản ứng khác sector chu kỳ không?

## 8. Kết luận làm việc

Trong repo hiện tại, hướng đáng theo là:

- không ép một LSTM khổng lồ học toàn thị trường ngay
- dùng `shared-market LSTM` như một bài kiểm tra khả năng học context
- giữ `sector / mini-group experts` là nơi tạo alpha chính
- chỉ nâng shared LSTM lên backbone khi nó chứng minh được breadth qua nhiều sector, không chỉ một case F&B

## 9. Overnight runner

Runner để treo máy cho watchlist whole-market là:

- [`run_whole_market_watchlist_overnight.sh`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_whole_market_watchlist_overnight.sh)

Script này sẽ:

- train `shared_vn100_w20`
- train `shared_vn100_w60`
- train 4 sector watchlist:
  - `Ngân hàng`
  - `Dịch vụ tài chính`
  - `Xây dựng và Vật liệu`
  - `Điện, nước & xăng dầu khí đốt`
- chạy committee giữa mỗi shared context và mỗi sector run
- xuất summary sáng hôm sau bằng:
  - [`summarize_whole_market_watchlist.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/research/summarize_whole_market_watchlist.py)
