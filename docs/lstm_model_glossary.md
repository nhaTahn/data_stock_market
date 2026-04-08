# LSTM Model Glossary

Tài liệu này giúp đọc đúng tên run, tên model, và các biến quan trọng trong repo.

## 1. Cách đọc tên run

Ví dụ:

```text
mini_bat_ong_san_g01_return_w20_pruned_v2
```

Ý nghĩa:

- `mini`: run tạo từ mini-group, không phải full sector
- `bat_ong_san`: slug của sector
- `g01`: group số 1 trong sector đó
- `return`: `target_mode=return`
- `w20`: `window_size=20`
- `pruned_v2`: tag do người chạy đặt để chỉ version feature set / experiment stage

Một ví dụ khác:

```text
sector_tpdouong_relscore_quantile_deep
```

- `sector`: pool cả sector
- `tpdouong`: slug của sector Thực phẩm và đồ uống
- `relscore`: train loss chính là `rel_score`
- `quantile`: có bật quantile family
- `deep`: run dùng ngân sách train sâu hơn, không phải smoke

## 2. Cách đọc tên model trong `metrics.json`, `predictions.csv`, `backtests`

### 2.1 Plain LSTM family

- `lstm`: model plain-LSTM mặc định của family.
- `lstm_seed_52`: plain-LSTM với random seed `52`.
- `lstm_ensemble`: trung bình dự báo của toàn bộ `lstm_seed_*` trong run.
- `lstm_best_by_val`: seed plain-LSTM có `val rel_score` cao nhất trong family.
- `lstm_top2_by_val`: trung bình dự báo của 2 seed plain-LSTM có `val rel_score` cao nhất.

Lưu ý quan trọng:

- `lstm` **không có nghĩa** là model tốt nhất.
- Trong code hiện tại, `lstm` là seed đầu tiên theo thứ tự sort của family plain-LSTM để giữ backward compatibility.

### 2.2 Sign-magnitude family

- `lstm_signmag`: model mặc định của family sign-magnitude.
- `lstm_signmag_seed_62`: sign-magnitude model với seed `62`.
- `lstm_signmag_ensemble`: trung bình dự báo của toàn bộ `lstm_signmag_seed_*`.
- `lstm_signmag_best_by_val`: seed sign-magnitude có `val rel_score` cao nhất trong family.
- `lstm_signmag_top2_by_val`: trung bình dự báo của 2 seed sign-magnitude tốt nhất theo validation.

Family này học 3 head nội bộ:

- `sign_prob`: xác suất chiều tăng/giảm
- `magnitude`: độ lớn biến động
- `signed_prediction`: dự báo cuối cùng có dấu

Trong report, repo chủ yếu dùng `signed_prediction`.

### 2.3 Quantile family

- `lstm_quantile`: model mặc định của family quantile.
- `lstm_quantile_seed_42`: quantile model với seed `42`.
- `lstm_quantile_ensemble`: trung bình dự báo `q50` của toàn bộ quantile seeds.
- `lstm_quantile_best_by_val`: seed quantile có `val rel_score` cao nhất khi chấm bằng `q50`.
- `lstm_quantile_top2_by_val`: trung bình `q50` của 2 quantile seeds tốt nhất theo validation.

Các cột phụ quan trọng trong `predictions.csv`:

- `prediction`: dự báo chính để chấm metric
- `prediction_q50`: median forecast của quantile model
- `prediction_q90`: upper quantile forecast
- `prediction_uncertainty`: `q90 - q50`

Trạng thái hiện tại:

- quantile family là **nhánh phụ**
- `prediction_uncertainty` hiện được dùng hữu ích hơn như **sidecar score** trong backtest, không phải model chính

### 2.4 Attention và event families

- `lstm_attention*`: family LSTM có self-attention sau backbone
- `lstm_event*`: family event-gated attention

Hai family này đang là nhánh phụ. Không nên mặc định coi chúng là baseline chính.

### 2.5 Baselines

- `linear_regression`: baseline hồi quy tuyến tính
- `arima`: baseline ARIMA

## 3. Cách đọc `best_by_val`, `ensemble`, `top2_by_val`

Ba tên này rất dễ bị hiểu nhầm:

- `best_by_val`: chọn một model bằng validation. Đây gần với cách chọn model thật khi chưa nhìn test.
- `ensemble`: lấy trung bình dự báo của toàn bộ seed trong family. Đây là model tổng hợp, không phải checkpoint train riêng.
- `top2_by_val`: chỉ ensemble 2 seed tốt nhất theo validation, dùng khi muốn bớt noise từ seed yếu.

Nếu `best_by_test` cao nhưng `best_by_val` thấp, run đó có ceiling tốt nhưng chưa ổn định để deploy.

## 4. Các biến cấu hình LSTM quan trọng

### Mục tiêu và dữ liệu

- `target_mode`: loại target. Thường dùng nhất là `return`, ngoài ra có `return_3d`, `return_5d`.
- `target_column`: cột target thật trong dataset sau khi map từ `target_mode`.
- `feature_columns`: danh sách feature dùng để train run đó.
- `sector`: tên sector nếu run theo sector.
- `stocks`: danh sách mã cụ thể nếu run theo mini-group hoặc run thủ công.

### Kiến trúc

- `window_size`: số phiên nhìn lùi cho mỗi sample sequence.
- `lstm_units`: số hidden units của backbone. Có thể là:
  - một số nguyên: 1 lớp LSTM
  - một list, ví dụ `[64, 32]`: 2 lớp LSTM
- `dropout`: dropout giữa các lớp hoặc sau pooling tùy family.

### Tối ưu

- `lr`: learning rate của Adam
- `loss`: loss train chính. Hiện khuyến nghị là `rel_score`
- `batch_size`: batch size
- `epochs`: số epoch tối đa
- `patience`: early stopping patience
- `lstm_seeds`: list seed chạy lặp để tạo family-level comparison

### Chuẩn hóa target

- `target_normalizer`: cách scale target theo local volatility, ví dụ `volatility_20`

Ý nghĩa thực tế:

- Khi bật `target_normalizer`, model thường học target đã scale theo regime
- Sau đó prediction sẽ được đưa về scale gốc trước khi chấm `rel_score`

### Sample weighting

- `sample_weight_mode`: cách tăng trọng số cho một số sample, ví dụ `magnitude`
- `sample_weight_strength`: độ mạnh của weighting
- `sample_weight_quantile`: quantile dùng làm mốc cho magnitude weighting
- `sample_weight_clip`: trần để tránh sample weight quá lớn

### Sign-magnitude weights

- `signmag_signed_loss_weight`: trọng số cho head dự báo signed final output
- `signmag_sign_loss_weight`: trọng số cho head phân loại dấu
- `signmag_magnitude_loss_weight`: trọng số cho head magnitude
- `signmag_log_magnitude`: nếu `true`, magnitude được học ở log-scale trước khi hoàn nguyên

### Attention

- `attention_enabled`: bật/tắt attention family
- `attention_heads`: số head trong multi-head attention
- `attention_key_dim`: kích thước key/query của attention

### Quantile

- `quantile_enabled`: bật/tắt quantile family

### Event

- `event_enabled`: bật/tắt event family
- `event_threshold`: ngưỡng để gán event sample
- `event_signed_loss_weight`
- `event_prob_loss_weight`
- `event_sign_loss_weight`
- `event_magnitude_loss_weight`
- `event_log_magnitude`

## 5. Cách đọc các cột report thường gặp

- `val`: metric trên validation split
- `test`: metric trên test split
- `DA`: directional accuracy
- `trade_count`: số lệnh trong backtest
- `coverage`: tỷ lệ số điểm test được trade
- `avg_strategy_return`: lợi nhuận trung bình mỗi lệnh
- `final_equity`: equity cuối nếu nhân chuỗi lợi nhuận

## 6. Những hiểu nhầm thường gặp

- `lstm` không phải lúc nào cũng là model tốt nhất.
- `best_by_val` không phải `best_by_test`.
- `ensemble` không phải model train lại từ đầu; nó là trung bình dự báo.
- Tag như `pruned_v2`, `deep`, `relscore`, `quantile` trong tên run là nhãn experiment, không phải parser chuẩn hóa cứng của repo.

## 7. File nên đọc cùng nhau

- `reports/core/config.json`: xem run đó dùng tham số gì
- `reports/core/metrics.json`: xem metric theo model
- `reports/core/family_selection_summary.json`: xem family nào chọn seed nào theo validation
- `reports/core/predictions.csv`: xem prediction theo từng model
- `reports/backtests/*.json`: xem model nào thật sự cho profile giao dịch tốt
