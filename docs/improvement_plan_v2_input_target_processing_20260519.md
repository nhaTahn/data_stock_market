# Cải Thiện Giai Đoạn 2: Input/Target Processing

Ngày: 2026-05-19

Cập nhật sau khi đã chạy xong: tail_loss, tailstress, stressaux, futurestress, marketaux, calibration probes.

## 1. Phân Tích Toán Học Của Kết Quả Hiện Tại

### 1.1 Quan sát chính từ data

| Candidate | rel_score | daily_max | days≥8% | pred/actual q90 |
| --- | ---: | ---: | ---: | ---: |
| stressaux_w20 (best) | 0.02477 | 9.44% | 7.7 | 0.166 |
| tail_loss | 0.02271 | 9.87% | 8.0 | 0.170 |
| tailstress | 0.02670 | 10.56% | 11.0 | 0.170 |
| base | 0.01667 | 9.51% | 13.0 | 0.134 |

**Pattern toán học quan trọng**: `pred/actual q90 ≈ 0.13–0.20` trên TẤT CẢ candidates.

### 1.2 Ý nghĩa của under-amplification

Model predict magnitude **~5–7 lần nhỏ hơn** thực tế ở q90.

Phân tích lý do:

**Lý do 1: Loss landscape của rel_score thúc đẩy shrinkage**

Khi DA ≈ 50% (signal yếu), gradient của rel_score loss tối ưu khi `|p|` nhỏ:

```
∂loss/∂|p| > 0  khi sign(p) ≠ sign(a)  (đa số samples)
∂loss/∂|p| < 0  khi sign(p) = sign(a)  (thiểu số samples khi DA<50%)
```

Net gradient → giảm |p|. Đây là behavior **đúng toán học** với metric này.

**Lý do 2: Calibration không sửa được vì sai source**

Calibration tăng `|p|` bằng cách `p_new = scale * p_old`. Nhưng:
- Khi đúng direction: `|p|` tăng → error giảm → tốt
- Khi sai direction: `|p|` tăng → error tăng → tệ

Net effect phụ thuộc DA. Với DA=50%, calibration là **trade-off zero-sum** ở mức tổng thể, nhưng:
- q50(error) có thể cải thiện (dominate bởi đúng direction samples)
- q90(error) chắc chắn xấu đi (dominate bởi sai direction × big move)

→ **Đây chính xác là pattern bạn quan sát**: rel_score tăng nhưng spike xấu hơn.

### 1.3 Kết luận toán học

**Bottleneck thật sự không phải scale**, mà là:

1. **Direction signal yếu** ở những ngày market move lớn (tail days).
2. **Magnitude conditional on correct direction** đã ở gần optimal cho features hiện tại.
3. **Tail days ≠ predictable days**: model không có thông tin để biết "ngày mai sẽ có move ±8%".

→ Hướng đúng: **giảm exposure ở tail days một cách selective**, không phải tăng amplitude tổng thể.

---

## 2. Đánh Giá Các Hướng Bạn Đề Xuất

### 2.1 Train target dạng market-relative residual — ✅ ĐÚNG TOÁN HỌC

Ý tưởng:
```
target_residual[i,t] = target_next_return[i,t] - market_proxy_return[t]
```

**Tại sao đúng**:
- Market component giải thích ~30–50% variance của individual stock returns.
- Tách market component → model chỉ cần học stock-specific signal (idiosyncratic alpha).
- Residual có volatility thấp hơn → q90(|target|) giảm → loss(base) giảm → rel_score "easier" về mặt scale.

**Vấn đề toán học cần xử lý**:

Khi report rel_score, có 2 lựa chọn:
- **Option A (residual evaluation)**: tính rel_score trên residual space.
  - Pro: consistent với training target.
  - Con: rel_score không so sánh được với baseline cũ.
- **Option B (raw evaluation)**: reconstruct prediction = residual_pred + market_pred, evaluate trên raw return.
  - Pro: comparable với existing baselines.
  - Con: cần predict market return separately, hoặc dùng known market_proxy_return ở t (leak nếu không cẩn thận).

**Đề xuất**: Option B với constraint:
```
final_prediction[i,t] = market_proxy_predicted[t] + alpha_predicted[i,t]
```

Trong đó `market_proxy_predicted[t]` được predict từ market features only (không dùng stock i features → no leak).

**Pitfall cần tránh**:
- Nếu dùng `market_proxy_return[t]` thực tế làm feature input → leak (vì market return là average of all stock returns, bao gồm stock i).
- Phải dùng `market_proxy_return[t-1]` hoặc lagged features only.

### 2.2 Market component head, output cuối là return — ✅ ĐÚNG VÀ AN TOÀN HỠN

Architecture đề xuất:

```
Inputs: [stock_features, market_features (lagged)]
       ↓
LSTM backbone
       ↓
   ┌──────────────┬─────────────────┐
Stock alpha    Market return    Sign confidence
head           head             head
   ↓              ↓                  ↓
alpha_pred     market_pred         conf
       ↓
final_pred = market_pred + alpha_pred * conf
```

**Loss design**:
```
loss = w1 * rel_score(actual, final_pred)
     + w2 * MSE(market_actual, market_pred)
     + w3 * BCE(sign(actual), sign(final_pred))
```

`market_actual[t] = mean(actual[*,t])` (cross-sectional mean of returns at day t).

**Tại sao tốt hơn auxiliary head trước đây**:

- `stressaux` predict `market_negative_ratio` → thông tin scale-only, không trực tiếp giúp prediction.
- `marketaux` predict `future_market_abs_return_q90` → leak future, fail nhanh trên screen.
- `market_return_head` đề xuất predict `market_actual[t] = mean over universe`:
  - Không leak (chỉ dùng market features lagged).
  - **Trực tiếp tham gia vào final prediction** (không phải auxiliary).
  - Nếu predict tốt → giảm |error| trên cả tail days lẫn normal days.

**Lý do toán học**:
```
error = actual - final_pred
      = actual - market_pred - alpha_pred * conf
      = (market_actual - market_pred) + (residual_actual - alpha_pred * conf)
      = market_error + alpha_error
```

Nếu `market_pred` đúng:
- Trên tail days (market move lớn): `market_error` ≈ 0 → tail error giảm mạnh.
- Trên normal days: `market_error` ≈ 0 → q50 error chủ yếu do alpha_error.

→ **Đây là cách giải quyết spike days đúng đắn nhất**: predict market component để "absorb" tail moves.

### 2.3 Point-in-time / Cross-sectional normalization rõ hơn — ✅ ĐÚNG

`multimarket_v1` hiện tại đã có:
- Rolling per-stock z-score (strict_past=True)
- Cross-sectional z-score và rank
- Market-level rolling z-score
- Calendar sin/cos

**Đánh giá implementation**:

✅ **Đúng**:
- `strict_past=True` đảm bảo `mean/std` chỉ dùng `t-w:t-1`, không leak.
- Cross-sectional z-score và rank tính trong `(market, Date)` group → portable.
- Market features có rolling z-score riêng.

⚠️ **Chưa tối ưu**:
- Cross-sectional z-score tại ngày `t` dùng features tại ngày `t` → **không leak target nhưng dùng same-day features của stock khác**.
- Đây là **point-in-time đúng** vì features đều đã có ở close của ngày `t`.
- Nhưng nếu features bao gồm `target_next_return` (như trong `add_tail_stress_features`) → leak.

**Cần kiểm tra**: `extra_feature_columns` trong probe có bao gồm `future_market_*` features không. Nhìn code → có. Đây là lý do `futurestress_w20` thất bại ở screen (43/71): feature đó là **future leak proxy**, model train tốt trên seed 52 (lucky alignment) nhưng không generalize.

### 2.4 Chạy lại 3 seeds với standard evaluation — ✅ CẦN THIẾT

Với 3 seeds (43, 52, 71) hiện tại:
- Seed 52: thường mạnh hơn → **có thể overfit feature/loss design qua seed selection**.
- Seed 43, 71: weaker → tránh false positive.

**Đề xuất standard protocol**:
- 3 seeds: 43, 52, 71 (hiện tại) hoặc thêm 5 seeds (42, 52, 62, 72, 82).
- Report mean + std + min của rel_score và daily_max.
- Promote candidate chỉ khi: `mean_rel_score > baseline_mean + std` AND `daily_max < 10%`.

---

## 3. Đề Xuất Pipeline Mới (Market-Relative Architecture)

### 3.1 Architecture: Two-Stream Market+Alpha

```
Stock features (lagged) ──┐
                          ├──→ Stock LSTM ──→ alpha_head (Dense, linear)
                          │                  ↓
                          │            alpha_prediction
                          │                  ↓
Market features (lagged) ─┼──→ Market LSTM ─→ market_head (Dense, linear)
                          │                  ↓
                          │            market_prediction
                          │                  ↓
                          └──────────────────┴──→ final_pred = market_pred + alpha_pred
```

**Lưu ý**: Market features chỉ chứa **market-level lagged** signals (không dùng same-day stock features cross-sectional). Điều này tránh leak.

Market features đề xuất:
```python
market_lagged = [
    "market_proxy_return_1_lag1",       # market return ngày trước
    "market_proxy_return_5",             # 5-day market return (đã lagged tự nhiên)
    "market_proxy_return_20",            # 20-day market return
    "market_volatility_20",              # market volatility
    "market_breadth_20",                 # market breadth
    "market_ad_ratio_20",                # advance/decline
    "market_negative_ratio_lag1",        # tail stress lagged
    "market_abs_return_q90_lag1",        # tail magnitude lagged
]
```

### 3.2 Loss Design

```
L_total = w_pred * RelScoreLoss(actual, final_pred)
        + w_market * Huber(market_actual, market_pred, delta=0.005)
        + w_alpha_l2 * mean(alpha_pred^2)        # regularize alpha to be small
```

Với:
- `w_pred = 1.0`
- `w_market = 0.5` (auxiliary supervision strong)
- `w_alpha_l2 = 0.05` (light shrinkage on alpha)

**Tại sao alpha L2 regularization**:
- Alpha là idiosyncratic component, kỳ vọng nhỏ (~0.5-1% daily).
- Khi không chắc, alpha → 0 là chiến lược an toàn.
- Phân biệt với output shrinkage post-hoc: ở đây shrinkage tích hợp trong training, nhất quán với loss.

### 3.3 Target Definition

```python
# Stock alpha target (idiosyncratic)
target_alpha[i, t] = target_next_return[i, t] - market_proxy_return_target[t]

# Market target (cross-sectional mean of next-day returns)
market_proxy_return_target[t] = mean_i(target_next_return[i, t])
```

**Important**: `market_proxy_return_target` là **mean of future returns**, không phải lagged. Nó là **legitimate target** cho market_head, không phải feature.

### 3.4 Inference

```python
final_pred[i, t] = market_pred[t] + alpha_pred[i, t]
```

`market_pred[t]` là **scalar per day** (shared across all stocks).
`alpha_pred[i, t]` là **per-stock idiosyncratic prediction**.

---

## 4. Phân Tích Tại Sao Hướng Này Giải Quyết Được Spike Days

### 4.1 Spike day = market-wide event

Trên VN, daily max ≥ 8% thường xảy ra khi:
- Market cả nước chạy mạnh (FED, geopolitics, COVID...).
- Cross-sectional dispersion tăng mạnh.
- |market_actual[t]| ≥ 3-4%.

→ Hầu hết spike error là do **market move**, không phải stock-specific move.

### 4.2 Nếu market_pred bắt được trend

Giả sử:
- Day t là spike day, `actual[i,t] = +9%` cho stock i.
- Market mean = +6%, stock-specific alpha = +3%.
- Model hiện tại predict `~+1.5%` → error = 7.5% → q90 spike.

Với two-stream:
- `market_pred[t] = +4.5%` (predict được 75% market move từ market features lagged).
- `alpha_pred[i,t] = +1.5%` (gần như bây giờ).
- `final_pred = +6%` → error = 3% → **không còn là spike**.

### 4.3 Nếu market_pred sai

Giả sử market_pred = -1% (sai dấu):
- `final_pred = -1% + 1.5% = +0.5%` → error = 8.5% → vẫn là spike.

→ **Critical**: market_head phải predict đúng dấu. Nếu sign accuracy của market_head < 60%, lợi ích bị bù trừ bởi sai market_pred.

**Kiểm tra cần làm trước**:
```python
# Baseline predictability of market return
market_returns = data.groupby('Date')['target_next_return'].mean()
market_features = ... # lagged market features

# Simple baseline: AR(1)
ar1_corr = market_returns.autocorr(lag=1)
print(f"Market AR(1): {ar1_corr}")

# Linear regression with lagged features
# If R² > 0.05 → worth pursuing
```

Nếu `market_returns` có AR(1) > 0.05 hoặc R² > 0.05 với lagged features → two-stream có triển vọng.
Nếu market_returns gần như random → two-stream không giúp được.

---

## 5. Kế Hoạch Thực Hiện Cụ Thể

### Bước 1: Diagnostic — Market predictability

Trước khi implement architecture mới, cần verify:

```python
# experiments/analysis/diagnose_market_predictability.py
# Output: 
#   - AR(1), AR(2), AR(5) of cross-sectional mean return
#   - R² của linear model với lagged market features
#   - R² split by tail days vs normal days
```

**Decision rule**:
- Market R² ≥ 0.05 → tiến hành two-stream.
- Market R² < 0.02 → bỏ hướng này, chuyển sang selective abstention.

### Bước 2: Single-stream baseline với residual target

Nhanh hơn implement, kiểm tra hypothesis:

```python
# Train target = residual = actual - market_proxy_return_target
target_residual = target_next_return - market_proxy_return  # group by Date

# Same architecture, same features as current best
# Evaluate on raw return space:
final_pred = predicted_residual + market_proxy_return_target  # use known mean

# Compare vs current baseline
```

**Note**: Khi evaluate, dùng `market_proxy_return_target[t]` thực tế (mean của actual returns ở t). Đây là **comparison upper bound** vì assume biết market mean. Nếu gain ≥ +0.005 rel_score → worth implementing market_head.

### Bước 3: Two-stream architecture

Nếu Bước 1+2 positive:

1. Implement `build_two_stream_model` trong `src/models/architectures/two_stream.py`.
2. Add market features pipeline (lagged-only) trong `load_frame`.
3. Add residual target computation trong `prepare_data`.
4. Train với 3 seeds (43, 52, 71), so sánh vs current best.

### Bước 4: Cross-market validation

Nếu two-stream work trên VN:

1. Áp dụng cùng pipeline lên JP, KR, US:
   - Tính market_proxy_return per market (đã có trong load_frame).
   - Tính residual target per market.
   - Inference: predict market + alpha per market.

2. Đánh giá:
   - rel_score per market.
   - Cross-market consistency: nếu cải thiện trên VN nhưng không cải thiện trên JP/KR → architecture bị overfit VN regime.

### Bước 5: Cải thiện market_head nếu cần

Nếu market_pred vẫn yếu:

**Option A: Multi-horizon market**
- Train 3 market heads cho 1d, 3d, 5d.
- Ensemble: market_pred = mean(1d, 3d_per_day, 5d_per_day).

**Option B: Heavier market features**
- Thêm leading indicators: VIX (US), JNK (high-yield spread), USDVND, gold, oil.
- Cross-market: dùng SPX overnight return làm leading indicator cho VN.

**Option C: Market regime classifier**
- Train classifier predict regime ∈ {tail_down, neutral, tail_up}.
- Conditional on regime, scale alpha differently.

---

## 6. Kiểm Tra Tính Đúng Đắn Toán Học Cho Multi-Market

### 6.1 Residual target portability

Test: Liệu residual target có scale-invariant qua markets?

```
residual_VN[i,t] = actual_VN[i,t] - mean_VN(t)
residual_JP[j,t] = actual_JP[j,t] - mean_JP(t)
```

✅ **Đúng**: Mỗi market có residual riêng (mean cross-sectional là baseline tự nhiên).
✅ Volatility của residual ≤ volatility của raw return → metric loss(base_residual) < loss(base_raw).
⚠️ Nhưng comparison rel_score(residual) vs rel_score(raw) **không trực tiếp**:
  - residual có loss(base) nhỏ hơn → cùng absolute error → rel_score residual sẽ thấp hơn.
  - Cần convert ngược về raw để compare.

### 6.2 Market_pred portability

Market features per market đã có:
- VN: vnindex_return, breadth_20, ad_ratio
- JP, US: tự tính cùng cách trong `load_frame`

**Issue**: Market behavior khác nhau:
- VN: bull-biased, retail-driven, momentum
- JP: institutional, mean-reverting
- US: sector rotation, complex

→ Một market_head trained trên VN có thể không transfer sang JP/US.

**Giải pháp**: Train market_head per market, hoặc thêm `market_id` embedding cho market_head.

### 6.3 Alpha portability

Alpha = residual return = stock-specific signal.
Alpha distribution:
- VN: cross-sectional std ~1.5% daily
- JP: ~1.0% daily
- US: ~1.2% daily

→ Khác scale. Cần normalize alpha target:

```
alpha_normalized[i,t,m] = alpha[i,t,m] / cross_sectional_std_alpha[m, t-w:t-1]
```

Inference inverse:
```
alpha_raw_pred = alpha_normalized_pred * cross_sectional_std_alpha
```

→ alpha_head learn portable representation.

### 6.4 Tổng hợp portability check

| Component | Portable? | Cách normalize |
| --- | --- | --- |
| Stock features | ✅ Đã ratio-based | rolling z-score (multimarket_v1) |
| Market features | ✅ Per market | rolling z-score |
| Cross-sectional features | ✅ Per market | rank/z-score within market |
| Calendar | ✅ | sin/cos |
| Target market_pred | ⚠️ Cần per-market scaling | divide by market volatility |
| Target alpha | ⚠️ Cần per-market scaling | divide by cross-sectional std of alpha |

---

## 7. Quyết Định Cuối Cùng

### 7.1 Đồng ý với hướng của bạn (input/target processing)

✅ Train target market-relative residual.
✅ Market component head trong architecture.
✅ Point-in-time / cross-sectional normalization rõ hơn.
✅ Standard 3-seed evaluation.

### 7.2 Bổ sung từ phân tích

1. **Verify market predictability trước**: nếu market mean return không predict được, không cần two-stream.
2. **Two-stream architecture** thay vì auxiliary head: market_pred phải là **component của final output**, không phải auxiliary loss.
3. **Alpha L2 regularization**: thay thế post-hoc shrinkage bằng training-time shrinkage.
4. **Cross-sectional std normalize cho alpha target**: cần thiết cho cross-market portability.
5. **Loại bỏ leak features**: kiểm tra `add_tail_stress_features` không bao gồm `future_market_*` trong training set.

### 7.3 KHÔNG nên làm tiếp

- ❌ Thêm auxiliary stress head (đã verify không giúp gốc rễ).
- ❌ Tăng tail_penalty_weight (đã thấy không cải thiện ổn định).
- ❌ Calibration/scaling output post-hoc (đã thấy trade-off zero-sum).

### 7.4 Mục tiêu cụ thể cho Bước 1+2

- VN val rel_score (3-seed mean): ≥ +0.030 (hiện best ~0.025).
- VN val daily_max p90: ≤ 8.5% (hiện best ~9.4%).
- VN val days≥8% (3-seed mean): ≤ 5 (hiện best ~7.7).
- Stable across 3 seeds: std ≤ 0.005 trên rel_score.

### 7.5 Files cần tạo

```
experiments/analysis/diagnose_market_predictability.py
experiments/training/run_residual_target_probe.py       # Bước 2
src/models/architectures/two_stream.py                  # Bước 3
experiments/training/run_two_stream_probe.py            # Bước 3
docs/market_predictability_readout.md                   # Output Bước 1
docs/residual_target_readout.md                         # Output Bước 2
docs/two_stream_readout.md                              # Output Bước 3
```

### 7.6 Trình tự ưu tiên

1. **Tuần này**: Bước 1 (diagnostic) + Bước 2 (residual target probe) — không cần architecture mới, chỉ thay target.
2. **Tuần sau**: Nếu Bước 2 positive → Bước 3 (two-stream).
3. **Tuần sau nữa**: Nếu Bước 3 positive trên VN → Bước 4 (cross-market test).
4. Nếu Bước 1 negative (market không predictable) → chuyển sang **selective abstention** thay vì two-stream:
   - Dùng filter signal hiện tại + abstain trên predicted-tail-stress days.
   - Hướng này đã có infrastructure (filter_signal.py, holding_period.py).
