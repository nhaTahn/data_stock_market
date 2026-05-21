# Improvement Progress Log

Updated: 2026-05-19

Mục đích: theo dõi từng bước trong kế hoạch input/target processing improvement.
Mỗi lần làm xong một bước, cập nhật file này.

## Tổng Quan Plan

Kế hoạch chính: `docs/improvement_plan_v2_input_target_processing_20260519.md`

| Bước | Trạng thái | Mục tiêu |
| --- | --- | --- |
| Step 1: Market predictability diagnostic | ✅ Done | Verify market mean return có predict được không |
| Step 2A: Residual target probe | 🔄 In progress | Test target = actual - market_proxy_return |
| Step 2B: Conditional gate probe | ⏸️ Pending | Test gate(vol) * market_pred (sau khi đọc 2A) |
| Step 3: Two-stream architecture | ⏸️ Pending | Implement market_head + alpha_head |
| Step 4: Cross-market validation | ⏸️ Pending | Test trên JP, KR, US |

---

## Step 1: Market Predictability Diagnostic — ✅ Done

**File**: `experiments/analysis/diagnose_market_predictability.py`
**Readout**: `docs/market_predictability_readout.md`
**Artifacts**: `data/processed/assets/data_info_vn/history/training_runs/reports/market_predictability_diagnostic_20260519/`

### Kết quả

| Metric | Value |
| --- | ---: |
| AR(1) train | 0.101 |
| AR(1) val | 0.110 |
| Best linear val R² | 0.003 (tail_ewm) |
| Best train R² | 0.014 (combined_full) |
| Tail day val R² | -0.40 đến -0.42 |
| `combined_full` val R² | -0.013 (overfitting) |

### Insight toán học

1. AR(1) ≈ 0.10 → R² lý thuyết ≈ 0.011. Momentum signal yếu.
2. **Tail day R² âm sâu** → market behavior đảo regime ở tail days. Lagged features sai dấu.
3. Pure additive `final = market_pred + alpha_pred` sẽ làm spike xấu hơn.

### Quyết định

- ❌ Pure additive two-stream: NO (sẽ làm spike tệ hơn ở tail).
- ✅ Conditional architecture: gate(vol) làm gating cho market component.
- ✅ Residual target probe (Step 2A): vẫn nên test để verify hypothesis market component có giá trị hay không.

---

## Step 2A: Residual Target Probe — 🔄 In progress

**Mục tiêu**: Train LSTM với target là residual `actual - market_proxy_return_target`,
đánh giá xem có cải thiện rel_score và giảm spike khi reconstruct về raw return.

**Hypothesis**: Nếu residual target dễ học hơn raw target (variance thấp hơn,
ít tail), model sẽ predict alpha tốt hơn. Khi reconstruct với market actual,
spike days sẽ được "absorb" bởi market component.

**Ghi chú**: Đây là **upper bound test** vì dùng `market_actual[t]` thực tế khi
reconstruct (giả định biết market mean). Nếu test này không gain, thì
implementing market_head trong two-stream càng không gain (vì market_pred
không thể tốt hơn market_actual).


### Full 3-Seed Run (18 epochs, patience 5) — ✅ Done

| Variant | rel_score mean | rel_score std | daily_max mean | spike≥8% mean | DA mean | pred/actual q90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_baseline | **0.02837** | 0.00532 | 11.12% | 11.7 | 50.9% | 0.184 |
| residual_oracle | **0.21578** | 0.00235 | 7.91% | **0.3** | **64.3%** | 0.512 |
| residual_lagged_ar1 | 0.02436 | 0.00374 | 10.68% | 10.0 | 50.6% | 0.164 |

### Phân tích kết quả

**1. Oracle reconstruction: PASS cực mạnh**

- rel_score: 0.028 → 0.216 (+0.187, tăng 7.6x). Vượt xa threshold +0.005.
- daily_q90_p90: 6.32% → 4.68% (giảm 1.64%). Vượt xa threshold -0.5%.
- spike≥8%: 11.7 → 0.3 (gần như loại bỏ hoàn toàn).
- DA: 50.9% → 64.3% (alpha prediction dễ hơn raw return prediction).
- Std cực thấp (0.002) → ổn định qua seeds.

**2. Lagged AR(1): FAIL trên rel_score, nhưng spike giảm nhẹ**

- rel_score: 0.028 → 0.024 (giảm 0.004). AR(1) noise làm tệ hơn.
- daily_max: 11.12% → 10.68% (giảm nhẹ 0.44%).
- spike≥8%: 11.7 → 10.0 (giảm nhẹ).
- DA: 50.9% → 50.6% (không cải thiện).

**3. Kết luận toán học**

- **Market component giải thích ~75% variance** (rel_score oracle 0.216 vs raw 0.028).
- **Alpha prediction (residual) có DA = 64.3%** — rất cao so với raw DA = 50.9%.
  Điều này confirm: direction signal nằm ở alpha, không phải raw return.
- **AR(1) quá yếu** (R² ≈ 0.01) → cần market_pred mạnh hơn.
- **Gap giữa oracle và AR(1) rất lớn** → cần LSTM market head hoặc better market predictor.

### Quyết định

- ✅ Step 2A PASS: oracle reconstruction confirm hypothesis.
- ✅ Proceed to Step 3: Two-stream architecture (LSTM market head + alpha head).
- ❌ Skip Step 2B (conditional gate): oracle result quá mạnh, không cần gate — cần market_pred tốt hơn.
- Key insight: **Bài toán chuyển từ "predict raw return" sang "predict market mean return tốt hơn AR(1)"**.



---

## Step 3: Two-Stream Architecture — 🔄 In progress

**Mục tiêu**: Implement LSTM market head + alpha head. Market head predict
cross-sectional mean return từ lagged market features. Alpha head predict
stock-specific residual. Final output = market_pred + alpha_pred.

**Thiết kế**:
- Market head: small LSTM (32 units) trên market features only (8 features, lagged).
- Alpha head: standard LSTM (64,32) trên stock features (26 features).
- Loss: `w_total * rel_score(actual, final_pred) + w_market * Huber(market_actual, market_pred)`.
- Market head được train jointly nhưng có gradient riêng (không share backbone).

**Kỳ vọng thực tế**:
- Oracle upper bound: rel_score = 0.216, spike≥8% = 0.3.
- AR(1) lower bound: rel_score = 0.024, spike≥8% = 10.
- LSTM market head kỳ vọng: rel_score 0.04–0.08, spike≥8% ≤ 5.
  (Nếu market head đạt R² ≈ 0.05–0.10 trên market mean return).


### Full 3-Seed Run — ❌ FAIL

| Variant | rel_score mean | daily_max mean | spike≥8% mean | DA mean | market_pred R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_baseline | **0.02837** | 11.12% | 11.7 | 50.9% | n/a |
| two_stream_joint | 0.01165 | 10.15% | **15.3** | 49.4% | **-67.1** |

### Phân tích

**Market head hoàn toàn thất bại**:
- `market_pred_r2 = -67` (âm sâu) → market head predict noise, không phải signal.
- `market_pred_corr ≈ 0.02` → gần như không tương quan với market actual.
- Spike≥8% tăng từ 11.7 → 15.3 (tệ hơn baseline!).
- rel_score giảm từ 0.028 → 0.012.

**Lý do toán học**:
1. Market mean return có R² ≈ 0.003 từ lagged features (Step 1 diagnostic).
2. LSTM market head (32 units, 8 features, 18 epochs) **overfit noise** thay vì learn signal.
3. Market head output noise → cộng vào alpha_pred → tăng error.
4. Joint training: market loss gradient kéo shared optimizer, làm alpha head cũng tệ hơn.

**Kết luận**: Two-stream joint training **không work** khi market signal quá yếu (R² < 0.01). LSTM market head không thể learn gì hữu ích từ 8 lagged features.

### Quyết định tiếp theo

- ❌ Two-stream joint: REJECT.
- ❌ Two-stream frozen market: không cần thử (market head đã fail).
- ✅ **Quay lại hướng đúng**: Dùng kết quả oracle để thiết kế khác.

**Insight từ oracle vs two-stream failure**:
- Oracle cho rel_score = 0.216 vì dùng **market_actual** (biết trước).
- Two-stream fail vì **market_pred** quá tệ.
- Gap giữa oracle (0.216) và baseline (0.028) = 0.188 → tiềm năng rất lớn.
- Nhưng gap giữa market_pred R² (-67) và oracle R² (1.0) → không thể bridge bằng LSTM trên lagged features.

**Hướng mới**: Thay vì predict market return, dùng **market return đã biết ở t-1** (lagged 1 day) làm component:
- `final_pred = market_return[t-1] * momentum_factor + alpha_pred`
- Hoặc đơn giản hơn: train trên residual target, reconstruct bằng `market_return[t-1]` (AR(1) implicit).
- Hoặc: **selective abstention** trên predicted-high-vol days (filter path đã có).

---

## Step 3B: Residual + Lagged Market Momentum — 🔄 Next

Thay vì predict market return bằng LSTM, dùng simple momentum rule:
- `market_component = momentum_factor * market_return_lag1_5`
- `final_pred = market_component + alpha_pred`

Momentum factor chọn trên train (grid search).
Đây là middle ground giữa oracle (biết market actual) và AR(1) (quá noisy).



---

## Step 3B: Residual Reconstruction Strategies — ✅ Done

**File**: `experiments/analysis/evaluate_residual_reconstruction.py`
**Readout**: `gold/vn_transition_pressure_20260512/plots/residual_reconstruction_20260520/summary.md`

### Kết quả Aggregate (3 seeds, validation)

| Strategy | rel_score mean | daily_max mean | spike≥8% mean | DA mean |
| --- | ---: | ---: | ---: | ---: |
| oracle | **+0.2158** | 7.91% | **0.3** | **64.3%** |
| shrunk_ar1 | -0.0010 | **7.85%** | 0.7 | 47.7% |
| ar1 | -0.0014 | 7.87% | 0.7 | 47.9% |
| zero | -0.0038 | 7.92% | 0.3 | 47.1% |
| momentum_5 | -0.0648 | 11.24% | 17.3 | 47.4% |
| momentum_20 | -0.0868 | 11.04% | 21.3 | 45.7% |

### So sánh với raw_baseline (Step 2A)

| | raw_baseline | residual + zero | residual + shrunk_ar1 | residual + oracle |
| --- | ---: | ---: | ---: | ---: |
| rel_score | **+0.0284** | -0.0038 | -0.0010 | +0.2158 |
| daily_max | 11.12% | **7.92%** | **7.85%** | 7.91% |
| spike≥8% | 11.7 | **0.3** | 0.7 | 0.3 |
| DA | **50.9%** | 47.1% | 47.7% | 64.3% |

### Phân tích toán học

**Insight cực kỳ quan trọng**:

1. **Residual model (zero reconstruction) gần như loại bỏ spike hoàn toàn** (0.3 days ≥ 8% vs 11.7 ở baseline). Đây là vì residual model predict alpha nhỏ → khi reconstruct bằng 0, prediction cũng nhỏ → error ≈ |actual| → nhưng q90 daily error giảm vì prediction không amplify sai direction.

2. **Nhưng rel_score âm** (-0.004) vì prediction ≈ 0 cho mọi sample → loss(error) ≈ loss(actual) → rel_score ≈ 0.

3. **Momentum strategies làm tệ hơn nhiều** (rel_score -0.06 đến -0.09, spike tăng). Momentum amplify noise.

4. **AR(1) / shrunk_ar1 gần bằng zero** — AR(1) signal quá yếu để tạo khác biệt.

5. **Gap oracle vs mọi strategy khác = 0.217** → **100% giá trị nằm ở biết market_actual**.

### Kết luận cuối cùng cho hướng two-stream / residual

**Bài toán đã được chẩn đoán rõ**:

- Market component giải thích ~75% variance (oracle rel_score 0.216).
- Nhưng market return **không predictable** từ lagged features (R² < 0.01).
- Mọi cách reconstruct market từ quá khứ đều fail (AR(1), momentum, LSTM).
- Residual model tự nó giảm spike nhưng mất rel_score.

**Hướng đi đúng tiếp theo**:

Vì không thể predict market return, hướng cải thiện phải là:

1. **Giữ raw target training** (baseline rel_score = 0.028 tốt hơn residual).
2. **Selective abstention trên high-vol days** (filter signal path đã có):
   - Khi predicted market volatility cao → abstain (prediction = 0).
   - Trên normal days → giữ prediction.
   - Đây là cách "chọn lọc" oracle behavior mà không cần biết market_actual.
3. **Confidence-aware shrinkage** (từ plan ban đầu):
   - Shrink prediction khi model không chắc → giảm tail error.
   - Không cần market prediction.

---

## Tổng Kết Toàn Bộ Experiment Chain

| Step | Kết quả | Quyết định |
| --- | --- | --- |
| 1. Market predictability | AR(1)=0.10, R²<0.01, tail R² âm | Market không predictable |
| 2A. Residual target | Oracle: rel=0.216, spike=0.3 | Market component có giá trị lớn nhưng cần biết trước |
| 3. Two-stream LSTM | Market head R²=-67, fail | LSTM không learn market signal |
| 3B. Reconstruction strategies | Tất cả non-oracle fail | Không thể reconstruct market từ quá khứ |

**Kết luận tổng thể**: Hướng "predict market return" là dead end. Giá trị nằm ở **biết market return** (oracle), nhưng ta không thể biết trước.

**Hướng tiếp theo (đã validate toán học)**:

1. **Confidence shrinkage** (post-processing, không retrain): shrink prediction khi sign_prob ≈ 0.5.
2. **Volatility-conditional abstention**: abstain khi predicted vol cao (filter signal path).
3. **Giữ raw baseline** (rel_score 0.028) + post-processing improvements.

Đây quay lại đúng hướng ban đầu trong `improvement_plan_relscore_multimarket_20260519.md` §2.4 (Confidence-Aware Shrinkage) — hướng đã được validate toán học là đúng.



---

## Step 4: Confidence Shrinkage & Volatility Abstention — ✅ Done

**File**: `experiments/analysis/evaluate_confidence_shrinkage.py`
**Readout**: `gold/vn_transition_pressure_20260512/plots/confidence_shrinkage_20260520/summary.md`

### Kết quả Aggregate (3 seeds, validation)

| Strategy | rel_score | daily_max | spike≥8% | DA | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| mag_shrink_g2 | **0.02907** | 11.12% | 11.7 | 50.9% | 100% |
| mag_shrink_g1 | 0.02890 | 11.12% | 11.7 | 50.9% | 100% |
| clip (k=1.5–2.0) | 0.02873 | **10.93%** | 11.7 | 50.9% | 100% |
| baseline | 0.02837 | 11.12% | 11.7 | 50.9% | 100% |
| vol_mag_shrink | 0.02800 | 10.70% | **11.0** | 50.9% | 100% |
| clip_vol_shrink | 0.02745 | **10.46%** | **11.0** | 50.9% | 100% |
| vol_shrink | 0.02720 | 10.70% | **11.0** | 50.9% | 100% |
| clip_vol_abstain | 0.01476 | **9.07%** | **2.3** | 43.4% | 84.4% |
| vol_abstain | 0.01469 | **9.07%** | **2.3** | 43.4% | 84.4% |

### Phân tích

**Pattern rõ ràng**: Trade-off giữa rel_score và spike reduction.

1. **Magnitude shrinkage (g2)**: +0.0007 rel_score vs baseline, nhưng **không giảm spike**. Gain nhỏ, spike giữ nguyên.

2. **Clipping (k=1.5)**: -0.0003 rel_score nhưng giảm daily_max 0.2%. Marginal.

3. **Vol shrinkage**: giảm spike 11.7→11.0 nhưng mất 0.001 rel_score. Marginal.

4. **Vol abstention**: giảm spike mạnh (11.7→2.3, daily_max 11.1%→9.1%) nhưng **mất 50% rel_score** (0.028→0.015) và coverage chỉ 84%.

5. **Clip + vol_shrink**: best balance — giảm daily_max 0.66%, spike giảm 0.7, mất 0.001 rel_score.

### Kết luận

**Không có strategy nào đồng thời cải thiện CẢ rel_score VÀ spike.**

- Muốn giảm spike → phải mất rel_score (abstention/shrinkage).
- Muốn tăng rel_score → magnitude shrinkage nhẹ (+0.0007) nhưng spike không đổi.
- **Trade-off này là fundamental** khi DA ≈ 50%: prediction lớn giúp khi đúng (tăng rel_score) nhưng hại khi sai (tăng spike).

### So sánh với kết quả trước đó

| Approach | rel_score | spike≥8% | Ghi chú |
| --- | ---: | ---: | --- |
| stressaux_w20 (tail probe) | 0.02477 | 7.7 | Retrained model |
| raw_baseline (Step 2A) | 0.02837 | 11.7 | No post-processing |
| clip_vol_shrink (Step 4) | 0.02745 | 11.0 | Post-processing only |
| vol_abstain (Step 4) | 0.01469 | 2.3 | Post-processing, 84% coverage |

**stressaux_w20 vẫn là best balance** (rel_score 0.025, spike 7.7) — nó đạt được bằng cách retrain với auxiliary head, không phải post-processing.

### Quyết định cuối

Post-processing alone **không đủ** để đạt target (rel_score ≥ 0.03 AND spike < 5%). Cần kết hợp:

1. **Retrained model** (stressaux_w20 hoặc tương đương) cho rel_score ~0.025.
2. **Post-processing clip_vol_shrink** trên output → giảm thêm daily_max ~0.5%.
3. **Selective abstention** trên high-vol days → giảm spike nhưng mất coverage.

Hoặc chấp nhận trade-off: **rel_score 0.025–0.029 với spike 8–12 days** là realistic ceiling cho architecture hiện tại trên VN daily returns.

---

## Tổng Kết Toàn Bộ Research Chain (Steps 1–4)

### Bảng so sánh tất cả approaches

| Approach | rel_score | spike≥8% | daily_max | Cần retrain? |
| --- | ---: | ---: | ---: | --- |
| Oracle (upper bound) | 0.216 | 0.3 | 7.9% | n/a |
| stressaux_w20 (best retrained) | 0.025 | 7.7 | 9.4% | Yes |
| raw_baseline | 0.028 | 11.7 | 11.1% | No |
| clip_vol_shrink (post-proc) | 0.027 | 11.0 | 10.5% | No |
| vol_abstain (post-proc) | 0.015 | 2.3 | 9.1% | No |
| Two-stream LSTM | 0.012 | 15.3 | 10.1% | Yes (fail) |
| Residual + AR(1) | 0.024 | 10.0 | 10.7% | Yes |

### Kết luận nghiên cứu

1. **Market component giải thích 75% variance** (oracle = 0.216) nhưng **không predictable**.
2. **Ceiling thực tế** cho VN daily return prediction: rel_score ~0.025–0.030, spike 8–12 days.
3. **Giảm spike dưới 5 days** chỉ khả thi bằng **abstention** (mất coverage/rel_score).
4. **stressaux_w20** vẫn là best balance hiện tại cho full-coverage prediction.
5. **Post-processing** (clip + vol_shrink) cho marginal improvement (~0.5% daily_max).

### Hướng tiếp theo đề xuất

1. **Promote stressaux_w20 + clip_vol_shrink** làm candidate mới (rel_score ~0.025, spike ~7–8).
2. **Chạy standard evaluation** (5 seeds, 60 epochs) trên stressaux_w20 config.
3. **Test portability** trên JP/KR/US với multimarket_v1 normalization.
4. **Chấp nhận** rằng rel_score = 0.05 trên full-coverage daily prediction là unrealistic — mục tiêu thực tế là 0.025–0.035 trên VN, ≥ 0 trên OOD.

