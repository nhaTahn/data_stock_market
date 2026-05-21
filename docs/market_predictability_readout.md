# Market Predictability Diagnostic Readout

Step 1 of input/target processing improvement plan.

Scope: VN train (<= 2020-03-31) and validation (2020-04-01 .. 2022-11-15).
Holdout/test is not used.

## Decision

- best train R^2: `0.0143` (combined_full)
- best val R^2: `0.0030` (tail_ewm)
- best |AR(k)|: `0.1095` (lag=1)
- recommendation: **proceed_two_stream**

## Autocorrelation of Target (Cross-Sectional Mean Return)

| Lag | Train AR(k) | Val AR(k) |
| ---: | ---: | ---: |
| 1 | `0.1011` | `0.1095` |
| 2 | `0.0699` | `0.0241` |
| 3 | `0.0459` | `0.0391` |
| 5 | `0.0099` | `0.0744` |
| 10 | `0.0644` | `0.0076` |

## Linear Regression R^2 (Lagged Market Features -> market_return_actual)

| Feature Set | n_train | n_val | R^2 train | R^2 val | RMSE val | R^2 tail val | R^2 normal val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar1_only | 2004 | 659 | `0.0045` | `-0.0016` | `0.01603` | `-0.4224` | `-0.0182` |
| ar1_ar2 | 2003 | 659 | `0.0059` | `-0.0007` | `0.01602` | `-0.4122` | `-0.0219` |
| rolling_returns | 1995 | 659 | `0.0022` | `0.0009` | `0.01601` | `-0.4093` | `-0.0206` |
| vol_only | 1995 | 659 | `0.0014` | `-0.0073` | `0.01607` | `-0.3971` | `-0.0413` |
| tail_only | 2004 | 659 | `0.0059` | `0.0026` | `0.01599` | `-0.4170` | `-0.0137` |
| tail_ewm | 2004 | 659 | `0.0055` | `0.0030` | `0.01599` | `-0.4094` | `-0.0169` |
| combined_minimal | 1995 | 659 | `0.0059` | `-0.0063` | `0.01606` | `-0.4134` | `-0.0311` |
| combined_full | 1995 | 659 | `0.0143` | `-0.0134` | `0.01612` | `-0.4073` | `-0.0468` |

## Top Lagged Features by Train Pearson Correlation

| Feature | Pearson | Spearman | n |
| --- | ---: | ---: | ---: |
| `market_q90_today_lag1` | `0.0739` | `0.0486` | 2004 |
| `market_negative_ratio_ewm5_lag1` | `-0.0721` | `-0.0435` | 2005 |
| `market_return_today_lag1` | `0.0670` | `0.0297` | 2004 |
| `market_negative_ratio_today_lag1` | `-0.0590` | `-0.0234` | 2005 |
| `market_breadth_today_lag1` | `0.0579` | `0.0308` | 2005 |
| `market_return_lag1_20` | `0.0443` | `0.0474` | 1995 |
| `market_return_today_lag2` | `0.0434` | `0.0255` | 2003 |
| `market_q10_today_lag1` | `0.0431` | `0.0118` | 2004 |
| `market_abs_q90_today_lag1` | `0.0404` | `0.0706` | 2004 |
| `market_volatility_lag1_20` | `-0.0378` | `-0.0002` | 1995 |
| `market_return_lag1_5` | `0.0366` | `0.0221` | 2002 |
| `market_abs_q90_ewm5_lag1` | `0.0273` | `0.0624` | 2004 |

## Interpretation

- Market return is sufficiently predictable from lagged features.
- Two-stream architecture (market_pred + alpha_pred) is worth implementing.
- Proceed to Step 2 (residual target probe).


## Phân Tích Sâu (Cập Nhật Sau Khi Đọc Số)

### Mâu thuẫn quan trọng

Decision tự động khuyến nghị `proceed_two_stream` vì AR(1) val = 0.1095 ≥ 0.05.

**Nhưng số thực tế cho thấy bức tranh khác**:

| Insight | Số |
| --- | ---: |
| AR(1) train | 0.101 |
| AR(1) val | 0.110 |
| R^2 từ AR(1) (lý thuyết) | 0.011 |
| Best linear val R^2 (mọi feature set) | 0.003 |
| `combined_full` val R^2 | **-0.013** (overfitting) |
| Tail day val R^2 (mọi feature set) | **-0.40 đến -0.42** |

### Diễn giải toán học

1. **AR(1) ≈ 0.10 chỉ giải thích ~1% variance**, không phải 10%.
   `R^2_AR1 = autocorr^2 = 0.0121`. Đây là upper bound cho variance reduction từ momentum signal.

2. **Validation R^2 ≤ 0.003** cho mọi linear feature set:
   - Train R^2 cao nhất (combined_full = 0.0143) → val âm = overfitting nặng.
   - Linear model không robust capture market return.

3. **Tail day R^2 âm sâu (-0.40)**: 
   - Trên tail days, market mean return là **anti-predictable** với linear features.
   - Predict bằng lagged features sẽ **làm tệ hơn** baseline (mean prediction).
   - Đây là regime change behavior: lagged features ngược dấu với actual ở tail.

### Hệ quả cho two-stream architecture

**Kịch bản tốt nhất** (LSTM market head capture full predictable signal):
- Market_pred giảm variance của error ~1% trên normal days.
- Market_pred làm **tệ hơn** trên tail days (do tail R^2 âm).

**Net effect dự kiến**:
- q50(|error|) giảm nhẹ (normal days dominate count).
- q90(|error|) **có thể tăng** (tail days dominate magnitude).
- → Cùng pattern bạn đã thấy với calibration: rel_score tăng nhưng spike xấu hơn.

### Quyết định chỉnh sửa

Decision tự động (proceed_two_stream) **quá lạc quan**. Decision đúng là:

→ **marginal_two_stream với conditional design**:

1. Two-stream có thể work, NHƯNG:
   - Market_head phải có cơ chế abstain/down-weight ở tail days.
   - Không thể là pure additive `final = market_pred + alpha_pred`.

2. Architecture đề xuất sửa lại:
   ```
   final_pred = gate(market_volatility) * market_pred + alpha_pred
   ```
   Với `gate ∈ [0, 1]`: 1 ở normal days, 0 ở high-vol/tail days.

3. **Hướng thay thế cần test song song**: 
   - **Conditional shrinkage** thay vì additive market_pred.
   - Predict `expected_market_volatility[t]` từ lagged features → khi predicted vol cao, shrink alpha_pred (vì tail days alpha cũng noisy hơn).
   - Đây là **non-additive use of market signal**, nhất quán với observation rằng tail prediction đảo dấu.

### Cập nhật trình tự

1. ✅ Step 1 (diagnostic): xong.
2. ✅ Step 2A (residual target probe): vẫn nên chạy — residual target ổn định toán học.
3. ⚠️ Step 2B (NEW): conditional gate probe. Test gate(vol) trước khi commit two-stream.
4. ⏸️ Step 3 (two-stream): chỉ làm sau khi 2A và 2B positive.
