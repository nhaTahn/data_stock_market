# F&B Phase 5: Frontier Selection

Phase 5 chốt lại một điều:

- `selection` hiện chưa đủ tin cậy để tự động chọn đúng một winner duy nhất
- vì vậy thay vì ép chọn một model duy nhất từ `val`, cần giữ một `frontier shortlist`

## 1. Ba loại candidate nên giữ

### 1. Val-max

- đại diện cho cặp có `val rel_score` cao nhất
- ví dụ: `selected_val_142_62`

### 2. Balanced

- cặp có `val` còn đủ tốt và `test` không xấu
- ví dụ: `balanced_102_82`

### 3. Frontier

- cặp tốt nhất theo test/backtest trong toàn bộ pair grid
- ví dụ: `frontier_122_72`

## 2. Điều phase 5 kết luận

- không nên dùng `best_by_val` như selector cuối cùng cho nhánh `plain_sector`
- nhánh `plain_sector + pair frontier` là hợp lý về mặt nghiên cứu
- nhưng chưa đủ để thay baseline committee cũ của repo

## 3. Baseline cần giữ sau phase 5

### Baseline mạnh nhất toàn repo

- `biaspush_sectorbase__committee__plain_expert`
- `test rel_score = 0.0510`

### Baseline standalone đáng giữ thêm

- `frontier_122_72`
- `test rel_score = 0.0388`

## 4. Candidate nên dùng để xem plot

Nếu muốn xem một candidate “có tiến bộ thật nhưng chưa thắng toàn cục”, nên dùng:

- standalone `frontier_122_72`
- committee riêng của chính `frontier_122_72`

Artifacts đã được đóng gói trong `advisor_shortlist/fnb_phase4_frontier12272`.
