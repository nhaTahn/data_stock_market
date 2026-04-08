# Báo cáo Phân tích Tương quan Feature — Thị trường Chứng khoán VN
**Ngày tạo:** 2026-04-08  
**Script sinh báo cáo:** `scripts/analyze_feature_correlation.py`  
**Dữ liệu nguồn:** `data/processed/assets/data_info_vn/history/vn_gold_recommended.csv`  
**Ngưỡng phân tích:** |Pearson correlation| ≥ 0.75  
**Output heatmaps:** `data/processed/assets/data_info_vn/history/training_runs/feature_correlation/<sector>/`

---

## 1. Tổng quan

Phân tích được thực hiện trên **16 sector**, **93 mã** thuộc chỉ số HOSE/VN30 mở rộng, với **28 features kỹ thuật** (sau khi loại macro context).

| Chỉ số | Giá trị |
|---|---|
| Số sector phân tích | 16 |
| Tổng số mã | 93 |
| Feature pool ban đầu | 28 features |
| Features được giữ lại (consensus) | **24 features** |
| Features bị loại (redundant) | **4 features** |
| Ngưỡng correlation | |Pearson| ≥ 0.75 |

### Cách đọc đúng báo cáo này

- Báo cáo này đo **redundancy giữa feature với feature**, không đo trực tiếp **signal của feature với target**.
- Script hiện tính correlation trên **toàn bộ train + val + test** để mô tả cấu trúc dữ liệu; vì vậy đây là tài liệu chẩn đoán, không phải bằng chứng out-of-sample.
- Bộ `24 features` trong báo cáo là **consensus candidate pool**. Khi train thật, từng sector/mini-group vẫn có thể dùng **một tập con** từ pool này.

---

## 2. Features bị loại do Redundant (toàn thị trường)

Bốn features sau **nhất quán bị redundant** trên tất cả 16 sector:

| Feature bị loại | Trùng với | Corr trung bình | Lý do |
|---|---|---|---|
| `atr_gap` | `volatility_20` | **+0.85 – +0.90** | Cả hai đo biến động giá theo ATR/Std, tương đồng gần như hoàn toàn |
| `ma_20_gap` | `momentum_20`, `rsi_14`, `bb_position` | **+0.77 – +0.85** | MA-gap 20 ngày mang cùng signal với momentum trung hạn và oscillator |
| `volume_zscore_20` | `volume_ratio_20` | **+0.84 – +0.92** | Hai cách normalize volume cùng một window — kết quả gần như đồng nhất |
| `bb_position` | `rsi_14` | **+0.84 – +0.86** | Cả hai đo vị trí giá trong vùng overbought/oversold theo cách khác nhau nhưng kết quả tương đương |

> **Kết luận:** Đưa cả 4 features này vào LSTM không thêm thông tin mới mà tăng nguy cơ overfitting do multicollinearity.

---

## 3. Feature Set Được Khuyến Nghị (24 features — dùng cho mọi sector)

```
vwap_gap, bb_width, volatility_20, gap_open, intraday_return,
close_position, obv_change, momentum_5, momentum_20, above_ma_200,
lower_shadow, upper_shadow, ma_200_gap, volume_ratio_20, rsi_14,
macd_hist, rolling_max_20_gap, wyckoff_phase_60d, effort_result_ratio,
buying_pressure, selling_pressure, vwap_gap_20*, alpha_sector*,
day_of_week
```

> `*`: `vwap_gap_20` có thể bị drop thêm ở một số sector nhỏ (Dầu khí, Bán lẻ). `alpha_sector` cần có `sector` column trong data.

---

## 4. Phân tích theo Nhóm Semantic

### 4.1 Nhóm bị redundant nặng (cần cắt)

| Nhóm | Features trong nhóm | Corr nội tại max | Quyết định |
|---|---|---|---|
| **Volatility** | `volatility_20`, `atr_gap` | **0.85 – 0.90** | Drop `atr_gap`, giữ `volatility_20` |
| **Volume ratio** | `volume_ratio_20`, `volume_zscore_20` | **0.84 – 0.92** | Drop `volume_zscore_20`, giữ `volume_ratio_20` |
| **MA/Oscillator overlap** | `ma_20_gap`, `rsi_14`, `bb_position` | **0.77 – 0.85** | Drop `ma_20_gap` + `bb_position`, giữ `rsi_14` |

### 4.2 Nhóm an toàn (giữ nguyên)

| Nhóm | Features | Corr nội tại max | Ghi chú |
|---|---|---|---|
| **Price shape** | `close_position`, `upper_shadow`, `lower_shadow` | ≤ 0.35 | Mỗi cái đo khía cạnh khác nhau của nến |
| **Momentum** | `momentum_5`, `momentum_20` | ≤ 0.50 | Khác window, bổ sung nhau |
| **MA long-term** | `ma_200_gap`, `rolling_max_20_gap` | ≤ 0.45 | Khác nhau về ý nghĩa xu hướng |
| **Bollinger** | `bb_width` (chỉ giữ 1) | — | Giữ `bb_width` (đo volatility regime), drop `bb_position` |
| **VWAP** | `vwap_gap`, `vwap_gap_20` | ~0 | Khác window, bổ sung nhau |
| **Wyckoff** | `effort_result_ratio`, `buying_pressure`, `selling_pressure`, `wyckoff_phase_60d` | ≤ 0.29 | Độc lập tốt |
| **MACD** | `macd_hist` (chỉ giữ 1) | — | Đại diện cho cả `macd`, `macd_signal` |

---

## 5. Chi tiết từng Sector

### 5.1 Tổng hợp số lượng

| Sector | Số mã | Features giữ | Features bỏ | Redundant pairs |
|---|---|---|---|---|
| Bất động sản | 18 | 24 | 4 | 8 |
| Ngân hàng | 14 | 24 | 4 | 8 |
| Thực phẩm và đồ uống | 10 | 24 | 4 | 9 |
| Dịch vụ tài chính | 9 | 23 | 5 | 9 |
| Xây dựng và Vật liệu | 9 | 23 | 5 | 9 |
| Điện, nước & xăng dầu khí đốt | 6 | 24 | 4 | 8 |
| Hàng & Dịch vụ Công nghiệp | 5 | 23 | 5 | 10 |
| Hóa chất | 5 | 24 | 4 | 9 |
| Tài nguyên Cơ bản | 4 | 23 | 5 | 13 |
| Bán lẻ | 3 | 22 | 6 | 12 |
| Công nghệ Thông tin | 2 | 24 | 4 | 8 |
| Du lịch và Giải trí | 2 | 23 | 5 | 9 |
| Dầu khí | 2 | 21 | 7 | 15 |
| Hàng cá nhân & Gia dụng | 2 | 24 | 4 | 9 |
| Bảo hiểm | 1 | 24 | 4 | 9 |
| Y tế | 1 | 24 | 4 | 8 |

### 5.2 Ghi chú đặc biệt theo sector

**Dầu khí (PLX, PVD)** — nhiều features nhất bị drop (7):
- Thêm: `ma_200_gap`, `rsi_14`, `vwap_gap_20` cũng bị loại
- Nguyên nhân: chỉ 2 mã, 1 mã (PVD) có pattern khác biệt mạnh

**Tài nguyên Cơ bản (HPG, HSG, NKG, PTB)** — 13 redundant pairs (cao nhất):
- Nhóm steel/metal có correlation cao hơn trung bình do chu kỳ giá nguyên liệu

**Bán lẻ (DGW, FRT, MWG)** — 6 features bị drop:
- Thêm `volatility_20` + `vwap_gap_20` cũng bị loại do đặc thù thanh khoản thấp

---

## 6. Top Redundant Pairs (Toàn Thị Trường)

Các cặp phổ biến nhất trên tất cả sector:

| Rank | Feature A | Feature B | Corr TB | Xuất hiện (sector) |
|---|---|---|---|---|
| 1 | `volume_ratio_20` | `volume_zscore_20` | +0.87 | 16/16 |
| 2 | `atr_gap` | `volatility_20` | +0.86 | 16/16 |
| 3 | `bb_position` | `rsi_14` | +0.85 | 16/16 |
| 4 | `momentum_20` | `ma_20_gap` | +0.85 | 16/16 |
| 5 | `ma_20_gap` | `rsi_14` | +0.83 | 16/16 |
| 6 | `ma_20_gap` | `bb_position` | +0.82 | 16/16 |
| 7 | `ma_20_gap` | `rolling_max_20_gap` | +0.78 | 14/16 |
| 8 | `momentum_5` | `ma_20_gap` | +0.77 | 13/16 |

---

## 7. Cách Diễn Giải Impact Với Training

### 7.1 Điều báo cáo này làm được và chưa làm được

- Correlation pruning giúp **giảm chồng lấp thông tin** giữa các feature.
- Correlation pruning **không tự động chứng minh** feature còn lại có alpha với `target_next_return`.
- Phần kết quả training bên dưới là **ảnh chụp một số run `pruned_v` đã có trong repo**, dùng để đánh giá tính thực dụng của bước pruning.

### 7.2 Quy tắc đọc leaderboard cho đúng

- `best by test` chỉ là **ceiling hậu nghiệm** để xem run đó có từng tạo ra model tốt hay không.
- `best by val` mới gần với cách **chọn model thật khi chưa nhìn test**.
- Nếu `best by test` cao nhưng `best by val` thấp, thì kết luận đúng là **run có tiềm năng nhưng selection chưa ổn định**, chứ chưa thể gọi là công thức đã thắng.

### 7.3 Snapshot các run pruned_v đáng chú ý

| Run | Nhóm mã | Best by val | Test (best by val) | Best by test | Test (best by test) | Nhận xét |
|---|---|---|---|---|---|---|
| BDS g01 v1 (`w15`) | KOS,DXG,NLG,DIG,TCH,VHM | `lstm` | +0.0198 | `lstm_ensemble` | +0.0290 | Pruning hữu ích, nhưng gap giữa chọn-val và ceiling vẫn còn |
| BDS g01 v2 (`w20`) | KOS,DXG,NLG,DIG,TCH,VHM | `lstm_seed_99` | +0.0086 | `lstm_signmag_seed_62` | **+0.0341** | Ceiling rất tốt, nhưng selection theo val vẫn chưa ổn định |
| Ngân hàng g02 v1 (`w15`) | VCB,TCB,CTG,BID,ACB,MBB | `lstm_signmag_seed_62` | +0.0048 | `lstm` | +0.0192 | Validation và test lệch mạnh |
| Ngân hàng g02 v2 (`w20`) | VCB,TCB,CTG,BID,ACB,MBB | `lstm_signmag_seed_99` | +0.0067 | `lstm_seed_52` | +0.0182 | Window dài hơn chưa tạo breakthrough rõ |
| Thực phẩm g03 v2 (`w20`) | DBC,VNM,ANV,SBT | `lstm_signmag_seed_99` | +0.0050 | `lstm_signmag_seed_62` | +0.0054 | Tín hiệu yếu nhưng đỡ lệch hơn ngân hàng |

### 7.4 Kết luận thực tế từ các run trên

- **Hợp lý**: dùng correlation pruning như một bước nén feature ban đầu là hợp lý.
- **Chưa đủ để kết luận công thức thắng**: câu "pruning + window=20 + signmag loss + 4 seeds là công thức cốt lõi" hiện đang nói quá mạnh.
- **Bằng chứng mạnh nhất** hiện chỉ nằm ở ceiling của nhóm Bất động sản `KOS,DXG,NLG,DIG,TCH,VHM`.
- **Điểm nghẽn còn lại** là stability của bước chọn model theo validation, không chỉ là chuyện bỏ feature redundant.

### 7.5 Nếu mục tiêu là kiểm tra feature có tín hiệu thật hay không

Cần thêm ít nhất 3 lớp kiểm tra ngoài correlation:

1. **Feature ↔ target linkage**
   - Spearman / IC giữa từng feature với `target_next_return`
   - tính riêng cho train, val, test thay vì gộp toàn bộ
2. **Rolling stability**
   - IC theo tháng hoặc theo quý để xem feature có đổi dấu theo regime không
3. **Ablation có kiểm soát**
   - giữ nguyên group mã, seed list, window
   - chỉ thay `feature_columns` để đo tác động thực của pruning

> Tóm lại: báo cáo này hợp lý như **diagnostic về redundancy**, nhưng chưa phải là bằng chứng đầy đủ rằng bộ feature hiện tại đã có signal tốt cho LSTM.

---

## 8. Files Đầu Ra

Mỗi sector có folder riêng tại:
```
data/processed/assets/data_info_vn/history/training_runs/feature_correlation/<sector>/
  ├── correlation_heatmap_full.png      # Heatmap toàn bộ 28 features
  ├── correlation_heatmap_pruned.png    # Heatmap sau khi drop redundant
  ├── missing_rate.png                  # Tỷ lệ NaN từng feature
  ├── feature_recommendation.json      # Feature set khuyến nghị + snippet config
  └── redundant_pairs.csv              # Danh sách cặp redundant chi tiết

data/processed/assets/data_info_vn/history/training_runs/feature_correlation/
  └── all_sectors_summary.csv          # Tổng hợp tất cả sector
```

---

## 9. Khuyến Nghị Hành Động

### Ngay lập tức
1. ✅ Đã xóa 4 features khỏi `default_features` trong `lstm_config.json`
2. ✅ Đã cập nhật `sector_features` cho 4 sector lớn
3. ✅ Đã có các run `pruned_v1` và `pruned_v2` để so sánh trực tiếp
4. 🔄 Nên bổ sung báo cáo `feature -> target` thay vì dừng ở `feature -> feature`

### Bước tiếp theo
- Chạy ablation cố định trên BDS g01: `old features` vs `pruned_v1` vs `pruned_v2`
- Với leaderboard, luôn đọc song song `best by val` và `best by test`
- Giữ `vwap_gap_20` như feature tùy chọn theo sector/mini-group, không ép thành mặc định toàn thị trường
- Sector nhỏ < 3 mã (Bảo hiểm, Y tế, CNTT, Dầu khí): ưu tiên model đơn mã hoặc cụm rất nhỏ

---

## 10. Tham chiếu

| File | Mô tả |
|---|---|
| [`scripts/analyze_feature_correlation.py`](scripts/analyze_feature_correlation.py) | Script sinh báo cáo này |
| [`configs/lstm_config.json`](configs/lstm_config.json) | Config đã cập nhật |
| [`src/utils/features.py`](src/utils/features.py) | Định nghĩa từng feature |
| [`src/evaluation/metric.py`](src/evaluation/metric.py) | Định nghĩa rel_score |
| [`docs/lstm_model_glossary.md`](docs/lstm_model_glossary.md) | Cách đọc tên model, seed, ensemble, best_by_val |
| [`docs/relscore_quantile_roadmap.md`](docs/relscore_quantile_roadmap.md) | Roadmap tổng thể |
