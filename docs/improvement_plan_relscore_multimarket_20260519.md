# Kế Hoạch Cải Thiện rel_score & Portability Đa Thị Trường

Ngày: 2026-05-19

## 1. Phân Tích Toán Học Của rel_score

### 1.1 Định nghĩa chính xác

```
loss(x) = q50(|x|) + 0.5 * q90(|x|)
rel_score = 1 - loss(error) / loss(base)
```

Trong đó `error = actual - prediction`, `base = actual`.

### 1.2 Phân rã error theo direction

Gọi `a = actual`, `p = prediction`. Chia thành 2 trường hợp:

**Trường hợp 1: Đúng direction** (`sign(p) = sign(a)`):

```
|error| = |a - p| = ||a| - |p||    (khi cùng dấu)
```

→ Error chỉ phụ thuộc vào sai lệch magnitude. Nếu `|p| ≈ |a|` thì `|error| ≈ 0`.

**Trường hợp 2: Sai direction** (`sign(p) ≠ sign(a)`):

```
|error| = |a - p| = |a| + |p|      (khi khác dấu)
```

→ Error luôn ≥ |a|. Đây là trường hợp **tệ hơn cả prediction = 0**.

### 1.3 Hệ quả toán học quan trọng

Gọi `α` = tỷ lệ sai direction, `E[|a|]` = kỳ vọng |actual|:

```
E[|error|] ≈ (1-α) * E[||a|-|p||] + α * E[|a|+|p|]
           = (1-α) * magnitude_error + α * (E[|a|] + E[|p|])
```

Với `α ≈ 0.515` (DA ≈ 48.5%), model hiện tại:
- 51.5% samples có `|error| = |a| + |p|` → **tệ hơn prediction=0**
- 48.5% samples có `|error| = ||a|-|p||` → tốt

**Kết luận toán học**: Khi DA < 50%, chiến lược tối ưu là **giảm |p| khi không chắc direction**. Đây chính xác là lý do `prediction = 0` (abstain) cho rel_score > 0 trên các samples sai direction.

### 1.4 Giới hạn lý thuyết của rel_score

Với daily return VN: `q50(|a|) ≈ 1.5%`, `q90(|a|) ≈ 4.0%`:

```
loss(base) = 1.5% + 0.5 * 4.0% = 3.5%
```

**Oracle bound** (prediction = actual):
```
rel_score_oracle = 1 - 0/3.5% = 1.0
```

**Zero prediction bound** (prediction = 0 cho mọi sample):
```
loss(error) = loss(actual) = loss(base)
rel_score_zero = 0.0
```

**Realistic bound** với DA=55%, magnitude RMSE=60% of |actual|:
```
loss(error) ≈ 0.45 * 0.6 * loss(base) + 0.55 * 1.4 * loss(base)
            ≈ (0.27 + 0.77) * loss(base) ≈ 1.04 * loss(base)
```

Sai! Tính lại chính xác hơn:

Với DA = 55%:
- 55% samples: |error| ≈ 0.6 * |a| (magnitude error khi đúng direction)
- 45% samples: |error| ≈ |a| + 0.6*|a| = 1.6 * |a| (sai direction)

```
q50(|error|) ≈ weighted_quantile ≈ 0.6 * q50(|a|) = 0.9%  (nếu đa số đúng)
q90(|error|) ≈ 1.6 * q90(|a|) = 6.4%  (tail bị dominate bởi sai direction)
loss(error) ≈ 0.9% + 0.5 * 6.4% = 4.1%
rel_score ≈ 1 - 4.1/3.5 = -0.17  (vẫn âm!)
```

**Insight quan trọng**: Ngay cả với DA=55%, nếu magnitude prediction quá lớn khi sai direction, rel_score vẫn âm. Metric này **phạt rất nặng** việc predict lớn sai hướng.

### 1.5 Chiến lược tối ưu cho rel_score

Từ phân tích trên, chiến lược tối ưu là:

1. **Shrink prediction toward zero** khi confidence thấp → giảm penalty khi sai direction.
2. **Predict lớn chỉ khi rất chắc direction** → tận dụng trường hợp đúng.
3. **Magnitude calibration**: khi đúng direction, `|p| ≈ |a|` cho error ≈ 0.

Công thức tối ưu:

```
p_optimal = confidence(direction) * E[|a| | features] * sign_prediction
```

Trong đó `confidence ∈ [0, 1]` là xác suất đúng direction.

Khi `confidence = 0.5` (random): `p_optimal = 0` (abstain).
Khi `confidence = 0.8`: `p_optimal = 0.6 * E[|a|] * sign`.

**Đây chính xác là logic của signmag model hiện tại** (`sign_prob * magnitude`), nhưng vấn đề là:
- `sign_prob` hiện tại không calibrated (DA < 50%).
- `magnitude` head bị couple với sign head qua shared backbone.

---

## 2. Đánh Giá Các Hướng Cải Thiện

### 2.1 Prediction Clipping — ✅ ĐÚNG TOÁN HỌC

```
p_clipped = clip(p, -k * σ_local, +k * σ_local)
```

Với `σ_local = volatility_20` (rolling 20-day std of returns).

**Tại sao đúng**: Giảm `|p|` khi `|p| > k*σ` → giảm penalty khi sai direction ở tail.
Với `k=2`: clip ở ~95th percentile của return distribution.

**Kỳ vọng**: Giảm q90(|error|) mà không ảnh hưởng q50 nhiều.

**Lưu ý**: Cần chọn `k` trên train, không phải validation.

### 2.2 Trimmed/Median Ensemble — ✅ ĐÚNG TOÁN HỌC

Mean ensemble: `p_ens = (1/K) Σ p_k`

Trimmed mean (bỏ min/max): giảm variance của ensemble prediction.

**Tại sao đúng**: Với K=5 seeds, 1 seed có thể overfit → predict quá lớn → tăng tail error. Trimmed mean loại bỏ outlier seed.

**Kỳ vọng**: Giảm q90(|error|) ~5-10%. Không cải thiện q50 nhiều.

**Median ensemble** còn robust hơn nhưng có thể mất signal ở tail.

### 2.3 Tách Sign Head Fine-tuning — ⚠️ CẦN THIẾT KẾ CẨN THẬN

Ý tưởng: Freeze backbone → fine-tune sign head riêng.

**Vấn đề toán học**:
- Sign head hiện dùng BCE loss với label = `1(actual > 0)`.
- BCE optimal output = `P(actual > 0 | features)`.
- Nhưng daily return gần symmetric quanh 0 → `P(up) ≈ 0.5` cho hầu hết samples.
- Fine-tuning sign head riêng sẽ **không cải thiện** nếu features không chứa directional signal.

**Kết luận**: Tách sign head chỉ có ý nghĩa nếu **thêm features mới có directional signal** (ví dụ: cross-sectional momentum rank, sector breadth). Nếu giữ nguyên features, sign head đã ở gần optimal rồi (DA ≈ 48.5% ≈ noise floor).

**Hướng đúng hơn**: Thay vì cải thiện DA, **calibrate confidence output** để model biết khi nào nó không chắc → shrink prediction.

### 2.4 Confidence-Aware Shrinkage — ✅ ĐÚNG VÀ MỚI

Thay vì cố cải thiện DA, dùng `sign_prob` hiện tại làm confidence:

```
p_shrunk = p * (2 * |sign_prob - 0.5|)^γ
```

Khi `sign_prob ≈ 0.5` (không chắc): shrink mạnh.
Khi `sign_prob ≈ 0` hoặc `≈ 1` (chắc): giữ nguyên.

`γ > 0` là hyperparameter (γ=1 linear, γ=2 quadratic shrinkage).

**Tại sao đúng**: Trực tiếp implement chiến lược tối ưu ở §1.5 mà không cần retrain.

### 2.5 RelScoreWeightedTailLoss Tuning — ⚠️ CÓ RỦI RO

Ý tưởng: Thêm penalty cho `|error| > threshold` trong training loss.

**Vấn đề toán học**:
- Tail penalty khuyến khích model predict gần 0 cho mọi sample (safe strategy).
- Nếu penalty quá mạnh → model collapse về `p ≈ 0` → rel_score = 0.
- Nếu penalty quá yếu → không có tác dụng.

**Kết luận**: Tail penalty trong training loss **không phải hướng tốt** vì:
1. Model đã được train với `rel_score` loss, vốn đã implicit penalize tail error (qua q90 term).
2. Thêm explicit tail penalty tạo conflict: model muốn predict lớn để giảm q50 error (khi đúng), nhưng bị phạt nếu predict lớn.
3. Post-processing (clipping, shrinkage) đạt cùng mục tiêu mà không cần retrain.

**Quyết định**: Bỏ hướng này. Dùng post-processing thay thế.

### 2.6 Multi-Horizon Ensemble — ✅ ĐÚNG NHƯNG CẦN ĐIỀU KIỆN

Train 3 models: 1d, 3d, 5d return targets.

**Toán học**:
```
target_1d = return[t→t+1]
target_3d = return[t→t+3] / 3  (annualized per day)
target_5d = return[t→t+5] / 5
```

Ensemble: `p_final = w1*p_1d + w2*p_3d + w3*p_5d`

**Tại sao có thể đúng**:
- `target_3d` và `target_5d` có signal-to-noise ratio cao hơn (mean reversion noise bị smooth).
- Ensemble giảm variance.

**Vấn đề**:
- Evaluation vẫn trên 1-day horizon → 3d/5d models predict "average daily return over 3/5 days", không phải "tomorrow's return".
- Nếu market có momentum (autocorrelation > 0): 3d/5d models capture trend tốt hơn.
- Nếu market có mean reversion (autocorrelation < 0): 3d/5d models sẽ **sai direction** cho 1-day.

**Điều kiện**: Chỉ hữu ích nếu VN daily returns có positive autocorrelation ở horizon 3-5 ngày. Cần kiểm tra trước khi implement.

### 2.7 Dedicated Sign + Magnitude Models — ❌ KHÔNG NÊN

Ý tưởng: Train 2 models riêng biệt.

**Vấn đề toán học**:
- Signmag hiện tại đã là architecture này (sign_prob + magnitude heads).
- Tách thành 2 models riêng **mất shared representation**.
- Sign model riêng sẽ vẫn đạt DA ≈ 48-49% vì features không chứa directional signal mạnh.
- Magnitude model riêng sẽ tốt hơn (không bị sign loss kéo), nhưng gain nhỏ.

**Kết luận**: Không tách. Thay vào đó, **giảm sign_loss_weight** xuống 0.05 hoặc 0 để backbone tập trung vào magnitude. Dùng sign_prob chỉ làm confidence indicator, không phải prediction component.

---

## 3. Chuẩn Hóa Dữ Liệu Cho Portability Đa Thị Trường

### 3.1 Đánh giá multimarket_v1 hiện tại

**Rolling per-stock z-score** (§1 trong doc):

```
x_roll = (f[t] - mean(f[t-w:t-1])) / (std(f[t-w:t-1]) + ε)
```

✅ **Đúng cho portability**: Mỗi stock tự normalize theo lịch sử riêng → không phụ thuộc scale tuyệt đối giữa markets.

⚠️ **Vấn đề**: 
- Window w=60 có thể quá dài cho features biến đổi nhanh (gap_open, intraday_return).
- z-score giả định distribution ổn định → sai trong regime change.
- **Đề xuất**: Dùng w=60 cho slow features (momentum_20, volatility_20), w=20 cho fast features (gap_open, intraday_return).

**Cross-sectional z-score** (§2):

```
x_csz = (f[i,t] - mean_universe(f[*,t])) / (std_universe(f[*,t]) + ε)
```

✅ **Đúng cho portability**: So sánh tương đối trong cùng universe → scale-invariant.

⚠️ **Vấn đề nghiêm trọng cho multi-market**:
- Universe `U[m,t]` được define **per market**. Đúng.
- Nhưng khi train joint (VN+JP+KR), cross-sectional stats phải tính **trong từng market riêng**, không gộp.
- Nếu gộp: VN stocks (volatility cao) sẽ luôn có z-score cao so với JP stocks (volatility thấp) → model học market bias thay vì stock signal.

**Cross-sectional rank** (§2):

```
x_rank = (rank(f[i,t]) - 1) / (|U[m,t]| - 1)
```

✅ **Rất tốt cho portability**: Rank ∈ [0,1] bất kể market, scale, hay distribution.

⚠️ **Vấn đề**:
- Universe size khác nhau: VN=93, JP=50, US=100. Rank resolution khác nhau.
- Rank mất thông tin về khoảng cách (stock rank 1 có thể cách rank 2 rất xa hoặc rất gần).
- **Đề xuất**: Dùng rank làm feature chính cho portability, giữ z-score làm feature phụ.

### 3.2 Vấn đề chưa giải quyết trong multimarket_v1

**Vấn đề 1: Target normalization**

Hiện tại: `target_normalizer = volatility_20`

```
y_normalized = target_next_return / max(volatility_20, floor)
```

✅ Đúng cho portability: return được scale theo local volatility → comparable across markets.

Nhưng **evaluation vẫn trên raw return**:
```
rel_score = 1 - loss(actual_raw - prediction_raw) / loss(actual_raw)
```

Nếu JP có volatility thấp hơn VN (q50(|a_JP|) ≈ 0.8% vs VN ≈ 1.5%):
```
loss(base_JP) ≈ 0.8% + 0.5*2.0% = 1.8%
```

→ Cùng absolute error, rel_score trên JP sẽ **tệ hơn** vì base nhỏ hơn.

**Đề xuất**: Khi report multi-market, dùng **volatility-adjusted rel_score**:
```
adjusted_error = (actual - prediction) / volatility_local
adjusted_base = actual / volatility_local
rel_score_adj = 1 - loss(adjusted_error) / loss(adjusted_base)
```

Hoặc đơn giản hơn: report rel_score per market, không gộp.

**Vấn đề 2: Market context features không portable**

Hiện tại:
- `vnindex_return` = equal-weight mean return of VN universe
- `a_d_ratio` = advancing/declining count in VN

Trên JP/US: cần tính tương tự nhưng từ universe riêng.

✅ Code hiện tại **đã xử lý đúng** (group by `market` column trong `load_frame`).

Nhưng **tên feature vẫn là `vnindex_return`** → confusing. Đề xuất rename thành `market_proxy_return` trong feature pipeline (giữ alias cho backward compat).

**Vấn đề 3: Sector features không có trên OOD**

Đây là vấn đề lớn nhất. Ablation cho thấy sector features chiếm ~40-60% edge:
- Bỏ `sector_rank`: rel_score giảm 44% (+0.00534 → +0.00301)
- Bỏ `sector_breadth`: rel_score giảm 51%
- Bỏ `sector_return_alpha`: rel_score giảm 78%

**Giải pháp portable**: Thay sector features bằng **data-driven cluster features**:

```
cluster_id[i,t] = assign_cluster(correlation_matrix[i, *, t-60:t-1])
cluster_return[c,t] = mean(return[j,t] for j in cluster c)
alpha_cluster[i,t] = return[i,t] - cluster_return[cluster_id[i,t], t]
cluster_momentum_rank[c,t] = rank(momentum_20 of cluster c)
```

Ưu điểm: Không cần sector metadata, tự adapt theo market structure.
Nhược điểm: Cluster không ổn định qua thời gian, cần rolling recompute.

### 3.3 Thiết kế chuẩn hóa đề xuất (multimarket_v2)

Dựa trên phân tích trên, đề xuất pipeline chuẩn hóa mới:

**Layer 1: Raw feature computation** (per stock, per market)
- Giữ nguyên tất cả technical features hiện tại.
- Tính market context per market (đã có).
- Bỏ sector features cứng, thay bằng cluster features (hoặc bỏ hẳn cho portable baseline).

**Layer 2: Per-stock rolling normalization**

| Feature group | Window | Transform |
| --- | --- | --- |
| Fast (gap_open, intraday_return, obv_change) | 20 | rolling z-score |
| Medium (momentum_5, volume_ratio_20, rsi_14, macd_hist) | 40 | rolling z-score |
| Slow (momentum_20, volatility_20, ma_200_gap, bb_width, wyckoff_phase_60d) | 60 | rolling z-score |
| Bounded [0,1] (close_position, buying_pressure, selling_pressure) | — | giữ nguyên (đã bounded) |
| Binary (above_ma_200) | — | giữ nguyên |

**Lý do phân window**: Features biến đổi nhanh cần window ngắn để z-score phản ánh "bất thường so với gần đây". Features chậm cần window dài để ổn định.

**Layer 3: Cross-sectional rank** (per market, per day)

Cho **tất cả** features sau Layer 2, thêm 1 view rank:

```
x_rank[i,t,j] = rank(x_layer2[i,t,j] among all i in market m at day t) / (N_m - 1)
```

Rank ∈ [0,1], hoàn toàn portable, không phụ thuộc scale.

**Layer 4: Market context rolling z-score** (per market)

```
market_return_z = (market_return - rolling_mean_60) / (rolling_std_60 + ε)
market_volatility_z = (market_vol - rolling_mean_60) / (rolling_std_60 + ε)
market_breadth_z = (breadth - rolling_mean_60) / (rolling_std_60 + ε)
```

**Layer 5: Calendar encoding**

```
day_sin = sin(2π * dow / 5)
day_cos = cos(2π * dow / 5)
```

**Layer 6: Target normalization**

```
y_train = target_next_return / max(volatility_20, floor)
```

Prediction output được inverse transform trước evaluation:
```
prediction_raw = prediction_normalized * max(volatility_20, floor)
```

### 3.4 Feature set portable đề xuất

**Tier 1: Core portable features** (dùng cho mọi market, không cần metadata ngoài)

```python
core_portable = [
    # Per-stock technicals (Layer 2 z-score + Layer 3 rank)
    "momentum_5",           # short-term momentum
    "momentum_20",          # medium-term momentum
    "volatility_20",        # realized volatility
    "volume_ratio_20",      # relative volume
    "close_position",       # intraday position (bounded, no transform)
    "upper_shadow",         # candle shape
    "lower_shadow",         # candle shape
    "gap_open",             # overnight gap
    "intraday_return",      # intraday move
    "bb_width",             # volatility regime
    "macd_hist",            # trend signal
    "rsi_14",               # oscillator
    "ma_200_gap",           # long-term trend
    "rolling_max_20_gap",   # distance from recent high
    "obv_change",           # volume-price confirmation
    "wyckoff_phase_60d",    # accumulation/distribution phase (bounded)
    "effort_result_ratio",  # volume vs range
    "buying_pressure",      # bounded [0,1]
    "selling_pressure",     # bounded [0,1]
]
```

**Tier 2: Market context** (tính per market, Layer 4 z-score)

```python
market_context = [
    "market_proxy_return",       # equal-weight universe return
    "market_proxy_volatility",   # rolling market volatility
    "market_breadth",            # advancing ratio
    "market_ad_ratio",           # advance/decline ratio
]
```

**Tier 3: Cross-sectional relative** (thay thế sector features)

```python
cross_sectional = [
    "momentum_20_cs_rank",       # rank of momentum within universe
    "volume_ratio_20_cs_rank",   # rank of volume within universe
    "volatility_20_cs_rank",     # rank of volatility within universe
    "momentum_5_cs_rank",        # rank of short momentum
    "rsi_14_cs_rank",            # rank of RSI
]
```

**Tier 4: Calendar**

```python
calendar = ["day_of_week_sin", "day_of_week_cos"]
```

**Tổng**: 19 (core) + 4 (market) + 5 (cross-sectional) + 2 (calendar) = **30 features**

So với hiện tại (26 features + sector features): tương đương về dimension, nhưng hoàn toàn portable.

### 3.5 Kiểm tra tính nhất quán toán học của pipeline

**Invariance test**: Nếu nhân tất cả prices của 1 market với hằng số `c`:
- `momentum_5` = ratio → không đổi ✅
- `volatility_20` = std(returns) → không đổi ✅
- `volume_ratio_20` = ratio → không đổi ✅
- `gap_open` = ratio → không đổi ✅
- `close_position` = (close-low)/(high-low) → không đổi ✅
- `ma_200_gap` = ratio → không đổi ✅
- `target_next_return` = ratio → không đổi ✅

→ Tất cả features đã là **scale-invariant** ở level raw. Rolling z-score thêm **time-invariance** (so sánh với chính mình trong quá khứ). Cross-sectional rank thêm **universe-invariance**.

**Currency test**: VN (VND), JP (JPY), US (USD) có scale giá rất khác:
- VN: giá ~10,000-200,000 VND
- JP: giá ~500-50,000 JPY  
- US: giá ~10-500 USD

Vì tất cả features là ratio/return-based → **không bị ảnh hưởng bởi currency**. ✅

**Volatility regime test**: VN daily vol ~2%, JP ~1.2%, US ~1.5%:
- Raw features: `volatility_20_VN ≈ 0.02`, `volatility_20_JP ≈ 0.012`
- Sau rolling z-score: cả hai ≈ 0 (so với chính mình) ✅
- Sau cross-sectional rank: cả hai ∈ [0,1] ✅

**Target normalization test**:
- VN: `target / vol_20 ≈ return / 0.02` → normalized target ~O(1)
- JP: `target / vol_20 ≈ return / 0.012` → normalized target ~O(1)
- Cùng scale → model học cùng mapping ✅

### 3.6 Vấn đề còn lại cần giải quyết

**1. Universe size effect trên cross-sectional rank**:
- VN: 93 stocks → rank resolution = 1/92 ≈ 1.1%
- JP: 50 stocks → rank resolution = 1/49 ≈ 2.0%
- US: 100 stocks → rank resolution = 1/99 ≈ 1.0%

Khác biệt nhỏ, chấp nhận được. Nếu muốn chính xác hơn, dùng **percentile rank** (đã là vậy trong formula).

**2. Trading calendar khác nhau**:
- VN: T+2 settlement, 5 ngày/tuần
- JP: T+2, 5 ngày/tuần, nhiều holidays hơn
- US: T+1 (mới), 5 ngày/tuần

→ `day_of_week` encoding vẫn đúng. Nhưng **holiday gaps** có thể tạo `gap_open` lớn bất thường. Rolling z-score sẽ handle (z-score cao = bất thường).

**3. Microstructure khác nhau**:
- VN: price limits ±7%, T+2, no short selling
- JP: price limits (varies), T+2
- US: no price limits, T+1, short selling

→ Price limits tạo **censored returns** ở VN/JP. Model trained trên VN sẽ không thấy returns > 7% → khi áp dụng US (có thể >10% daily), prediction sẽ underestimate.

**Giải pháp**: Target normalization by volatility đã partially handle (returns lớn ở US cũng đi kèm volatility cao). Nhưng cần **clip target** ở training để tránh outlier dominate:
```
target_clipped = clip(target_normalized, -4, +4)  # 4 sigma
```

---

## 4. Kế Hoạch Thực Hiện (Đã Chỉnh Sửa)

### Phase 1: Post-processing improvements (không retrain)

**1A. Confidence-aware shrinkage** ← MỚI, thay thế "tách sign head"

```python
sign_confidence = 2 * abs(sign_prob - 0.5)  # ∈ [0, 1]
p_shrunk = signed_prediction * sign_confidence ** gamma
```

Grid search `gamma ∈ [0.5, 1.0, 1.5, 2.0]` trên train predictions.

Kỳ vọng: rel_score +0.002–0.005 vì giảm penalty khi model không chắc direction.

**1B. Prediction clipping**

```python
p_clipped = np.clip(p_shrunk, -k * volatility_20, +k * volatility_20)
```

Grid search `k ∈ [2.0, 2.5, 3.0]` trên train.

Kỳ vọng: Giảm q90(|error|) thêm ~5-10%.

**1C. Trimmed mean ensemble**

Bỏ seed có val rel_score thấp nhất, mean 4 seeds còn lại.

Kỳ vọng: Giảm variance, rel_score +0.001.

**Tổng kỳ vọng Phase 1**: rel_score VN từ +0.005 → +0.010–0.015.

### Phase 2: Portable normalization (cần retrain)

**2A. Implement multimarket_v2 normalization**

- Adaptive window per feature group (20/40/60).
- Cross-sectional rank cho tất cả features.
- Market context rolling z-score.
- Calendar sin/cos.

**2B. A/B test trên VN**

```
Baseline: current config (normalization=none, sector features)
Variant A: multimarket_v2, giữ sector features
Variant B: multimarket_v2, portable feature set (no sector)
```

Acceptance criteria:
- Variant A: rel_score ≥ 95% of baseline (sector features vẫn có)
- Variant B: rel_score ≥ 70% of baseline (acceptable loss for portability)

**2C. OOD evaluation**

Nếu Variant B đạt criteria → chạy trên JP/KR/US.
Mục tiêu: rel_score ≥ 0 trên ít nhất 2/3 markets.

### Phase 3: Model improvements (nếu Phase 1+2 chưa đủ)

**3A. Giảm sign_loss_weight**

Từ 0.15 → 0.05 hoặc 0.0. Để backbone tập trung magnitude.
Sign_prob chỉ dùng làm confidence indicator trong post-processing.

**3B. Multi-market joint training**

Gộp VN + JP + KR (sau khi portable normalization đã validate).
Thêm `market_id` one-hot (3 dim) hoặc embedding.

**3C. Horizon check**

Kiểm tra autocorrelation structure:
```python
for lag in [1, 2, 3, 5]:
    print(f"AC(lag={lag}): {returns.autocorr(lag)}")
```

Nếu AC(3) > 0.02: multi-horizon ensemble có ý nghĩa.
Nếu AC(3) ≈ 0: không nên dùng multi-horizon.

---

## 5. Kỳ Vọng Thực Tế Về rel_score = 0.05

### 5.1 Tại sao 0.05 rất khó trên daily returns

Với daily return distribution điển hình:
- Signal-to-noise ratio ≈ 0.05–0.10 (Sharpe ratio hàng năm ~0.8–1.5)
- Optimal prediction chỉ giải thích ~0.5–1% variance của daily returns
- rel_score = 0.05 đòi hỏi giảm loss(error) 5% so với loss(base)

Tính ngược: cần `loss(error) = 0.95 * 3.5% = 3.325%`.
Hiện tại: `loss(error) ≈ 3.48%`.
Cần giảm: `3.48% - 3.325% = 0.155%` tuyệt đối.

Với q50(|error|) ≈ 1.48% và q90(|error|) ≈ 4.0%:
- Giảm q50 từ 1.48% → 1.38% (giảm 7%) VÀ
- Giảm q90 từ 4.0% → 3.7% (giảm 7.5%)

Đây là feasible nhưng đòi hỏi cải thiện **cả median lẫn tail** cùng lúc.

### 5.2 Điều kiện đạt rel_score ≥ 0.05

| Điều kiện | Hiện tại | Cần đạt | Khả thi? |
| --- | --- | --- | --- |
| DA (directional accuracy) | 48.5% | ≥ 52% | Khó trên daily |
| Magnitude RMSE / |actual| | ~100% | ≤ 85% | Feasible với shrinkage |
| Tail control (q90 error / q90 base) | ~100% | ≤ 90% | Feasible với clipping |
| Selective prediction (coverage) | 100% | 40-60% | Đã có filter path |

**Kết luận**: rel_score = 0.05 trên **toàn bộ universe** rất khó. Nhưng trên **filtered subset** (40-60% coverage) thì feasible vì:
- Filter chọn samples có confidence cao → DA tăng trên subset
- Abstain trên samples khó → không bị penalty

Hiện tại `move_top_train_ic_selected` đã đạt +0.0072 với 40.5% coverage.
Với confidence shrinkage + clipping trên filtered subset: kỳ vọng +0.015–0.025.

### 5.3 Mục tiêu đề xuất (thực tế)

| Metric | Mục tiêu VN | Mục tiêu OOD | Timeline |
| --- | --- | --- | --- |
| rel_score (full universe) | ≥ +0.015 | ≥ +0.005 | 6 tuần |
| rel_score (filtered 40%) | ≥ +0.030 | ≥ +0.015 | 6 tuần |
| abs(E) = loss(error) | < 3.3% | < market-specific base | 4 tuần |
| Mean daily IC | ≥ +0.05 | ≥ +0.03 | 4 tuần |
| DA | ≥ 50% (filtered) | ≥ 50% (filtered) | 6 tuần |

### 5.4 Về mục tiêu abs(E) < 3.5%

`abs(E) = loss(error) = q50(|error|) + 0.5 * q90(|error|)`

Trên VN hiện tại: ≈ 3.48% → **gần đạt**.

Trên OOD:
- JP: loss(base) ≈ 1.8% (volatility thấp) → abs(E) < 3.5% dễ đạt nhưng rel_score vẫn có thể âm.
- US: loss(base) ≈ 2.5% → abs(E) < 3.5% feasible.
- KR: loss(base) ≈ 2.8% → abs(E) < 3.5% feasible.

**Lưu ý**: abs(E) < 3.5% là điều kiện **cần** nhưng **chưa đủ** cho rel_score > 0. Cần abs(E) < loss(base) per market.

---

## 6. Tóm Tắt Quyết Định

| Hướng | Quyết định | Lý do toán học |
| --- | --- | --- |
| Prediction clipping | ✅ Làm | Giảm tail penalty khi sai direction |
| Confidence shrinkage | ✅ Làm (ưu tiên cao nhất) | Trực tiếp implement optimal strategy |
| Trimmed ensemble | ✅ Làm | Giảm variance, giảm tail |
| Tách sign head | ❌ Bỏ | DA bottleneck do features, không do architecture |
| Tail loss tuning | ❌ Bỏ | Post-processing đạt cùng mục tiêu, ít risk hơn |
| Multi-horizon | ⏸️ Chờ kiểm tra AC | Chỉ có ý nghĩa nếu positive autocorrelation |
| Dedicated sign+mag models | ❌ Bỏ | Mất shared representation, gain nhỏ |
| multimarket_v2 normalization | ✅ Làm | Cần thiết cho portability |
| Portable feature set (no sector) | ✅ Làm | Cần thiết cho OOD |
| Cross-sectional rank features | ✅ Làm | Thay thế sector features, portable |
| Data-driven cluster features | ⏸️ Phase sau | Phức tạp, chờ portable baseline |
| Giảm sign_loss_weight | ✅ Thử | Backbone tập trung magnitude |
| Multi-market joint training | ⏸️ Phase sau | Chờ portable normalization validate |
