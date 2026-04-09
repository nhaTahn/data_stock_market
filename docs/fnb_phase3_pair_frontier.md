# F&B Phase 3: Pair Frontier

Phase 3 không train thêm kiến trúc mới.

Mục tiêu:

- tách riêng vấn đề `representation` khỏi vấn đề `selection`
- đóng gói các cặp seed mạnh nhất từ `phase2_plain_sector_deepseeds`
- backtest và committee lại chúng như các candidate độc lập

## 1. Tại sao cần phase 3

Phase 2 cho thấy:

- `plain_sector_base` có tín hiệu
- nhưng pair được chọn theo `val` vẫn fail trên `test`
- trong khi vài pair khác lại đạt `test` khá tốt

Nghĩa là:

- nhánh `plain + sector features` chưa bị loại
- nhưng selection rule hiện tại chưa dùng được

## 2. Candidate frontier được đóng gói

Ba cặp được giữ:

1. `selected_val_142_62`
- cặp mà phase 2 đã chọn theo `val`

2. `balanced_102_82`
- cặp có `val` còn đủ tốt và `test` không quá tệ

3. `frontier_122_72`
- cặp tốt nhất theo `test` trong pair grid

## 3. Câu hỏi phase 3 cần trả lời

1. Cặp `frontier_122_72` có thật sự tốt hơn standalone baseline cũ không?
2. Khi ghép với `signmag sector-base`, cặp frontier có đưa committee trở lại gần vùng `0.05` không?
3. Nếu có, repo nên giữ hướng `plain_sector + pair frontier` hay vẫn quay về committee cũ?

## 4. Tiêu chí đọc kết quả

- nếu `frontier_122_72` standalone > `0.0343` thì plain-sector path vẫn đáng theo
- nếu committee từ frontier > `0.05` thì phase 3 thành công rõ
- nếu frontier chỉ tốt khi nhìn test nhưng committee không ổn, thì nhánh này chỉ mới là evidence nghiên cứu, chưa phải baseline mới
