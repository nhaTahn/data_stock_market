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



---

## Step 5: Heteroscedastic NLL Probe — ✅ Done

**File**: `experiments/training/run_hetero_nll_probe.py`
**Readout**: `gold/vn_transition_pressure_20260512/plots/hetero_nll_probe_20260521/summary.md`

### Kết quả Aggregate Validation (3 seeds)

| Variant | rel\_score | daily\_max | spike≥8% | DA | mean σ |
| --- | ---: | ---: | ---: | ---: | ---: |
| hetero\_combined | **+0.0372** | 11.23% | 13.7 | 51.2% | 2.37% |
| baseline (in-script) | +0.0195 | 9.71% | 7.0 | 50.4% | 0 |
| hetero\_nll | +0.0141 | 9.05% | **5.0** | 50.1% | 2.36% |

So sánh với stressaux\_w20 (production best): rel\_score +0.0248, spike 7.7.

### Phân tích

1. **hetero\_combined tăng rel\_score mạnh** (+0.0372 vs baseline +0.0195 vs stressaux +0.0248). Gain +0.012 so với stressaux. Đây là rel\_score cao nhất đạt được.

2. **Nhưng spike tăng** (13.7 vs 7.0 baseline, vs 7.7 stressaux). Combined loss khuyến khích predict lớn hơn (vì NLL reward calibrated amplitude) → spike tệ hơn.

3. **hetero\_nll thuần giảm spike** (5.0) nhưng mất rel\_score (0.014 < baseline 0.020). NLL thuần quá conservative → shrink prediction mạnh → mất signal.

4. **σ head học được** (mean σ ≈ 2.37% ≈ VN daily volatility). Nhưng ±2σ clip là no-op (2σ ≈ 4.74% > hầu hết predictions).

### Kết luận

- **hetero\_combined** là ứng viên mới cho rel\_score (best ever +0.037).
- Nhưng cần **σ-aware shrinkage** riêng để giảm spike mà không mất rel\_score:
  - Shrink prediction khi σ cao: `pred_shrunk = pred * min(1, k / sigma)`.
  - Hoặc selective coverage: chỉ trade khi σ thấp.

### Bước tiếp

Implement post-processing σ-aware shrinkage trên hetero\_combined predictions:
- Grid search: `k ∈ [0.01, 0.015, 0.02, 0.025, 0.03]`.
- Evaluate: rel\_score + spike days.
- Target: giữ rel\_score ≥ 0.030 VÀ spike ≤ 8.


---

## Step 5 (P1): Heteroscedastic σ Head + NLL Loss — ✅ Done

**Files**: `experiments/training/run_hetero_nll_probe.py`
**Artifacts**: `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_nll_probe_20260521/`
**Gold**: `gold/vn_transition_pressure_20260512/plots/hetero_nll_probe_20260521/`

### Kết quả Aggregate (3 seeds: 43, 52, 71 — 18 epochs, patience 5)

| Variant | rel_score mean ± std | daily_q90_max | spike≥8% | DA | pred/act q90 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| hetero_combined (0.7 rel + 0.3 NLL) | **0.0372 ± 0.0043** | 11.23% | 13.7 | 51.2% | 0.175 |
| baseline (rel_score_weighted_tail) | 0.0195 ± 0.0099 | 9.71% | 7.0 | 50.4% | 0.158 |
| hetero_nll (pure NLL) | 0.0141 ± 0.0059 | 9.05% | 5.0 | 50.1% | 0.141 |

Comparison với production anchor: stressaux_w20 = 0.0248 ± 0.007, spike 7.7.

### Phân tích

1. **hetero_combined gains +0.0124 vs stressaux_w20** trên rel_score, std thấp hơn (0.004 vs 0.007).
2. **Trade-off rõ ràng**: rel_score tăng mạnh nhưng spike tăng (13.7 vs 7.7) và daily_q90_max từ 9.44% → 11.23%.
3. **σ head học được** (mean_sigma_val ≈ 2.37%) nhưng simple clip ±2σ là no-op (mu nhỏ hơn nhiều so với σ).
4. **hetero_nll** giảm spike (5.0) nhưng rel_score thấp hơn — loss NLL thuần không giữ được signal.

### Kết luận P1

- ✅ hetero_combined là kiến trúc tốt hơn cho rel_score.
- ⚠️ Cần kiểm soát spike bằng σ-shrinkage (P2) — σ head có tín hiệu nhưng cần exploit đúng cách.
- ✅ Proceed to P2.

---

## Step 6 (P2): σ-Aware Shrinkage Post-Processing — ✅ Done

**Files**: `experiments/training/run_sigma_shrinkage_probe.py`
**Artifacts**: `data/processed/assets/data_info_vn/history/training_runs/reports/sigma_shrinkage_probe_20260521/`
**Gold**: `gold/vn_transition_pressure_20260512/plots/sigma_shrinkage_probe_20260521/`

### Kết quả Aggregate (3 seeds, val split)

| Rule | rel_score | daily_q90_max | spike≥8% | DA | ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ⭐ stressaux_w20 (prod) | 0.0248 | 9.44% | 7.7 | 50.7% | 0.166 | 100% |
| raw (hetero_combined) | **0.0372** | 11.23% | 13.7 | 51.2% | 0.175 | 100% |
| sigma_clip_k1.5 / k2.0 | 0.0372 | 11.23% | 13.7 | 51.2% | 0.175 | 100% |
| selective_strong (2.4% cov) | 0.0196 | 7.65% | 0.3 | 55.0% | 0.253 | 2.4% |
| abstain_sigma_top10 (79% cov) | 0.0108 | 8.31% | 1.3 | 50.1% | 0.160 | 79% |
| sigma_shrink_k1.0 | 0.0106 | 7.91% | 0.7 | 51.2% | 0.094 | 100% |
| shrink_k1.5 + abstain_top10 | 0.0103 | 7.63% | 0.3 | 50.1% | 0.108 | 79% |
| sigma_shrink_k1.5 | 0.0094 | 7.79% | 0.0 | 51.2% | 0.082 | 100% |
| sigma_shrink_k2.0+ | 0.0089–0.0086 | 7.77% | 0.0 | 51.2% | <0.08 | 100% |
| abstain_sigma_top25 (57% cov) | 0.0079 | 7.79% | 0.3 | 49.4% | 0.153 | 57% |

### Phân tích

**5 insight quan trọng**:

1. **sigma_clip không bind**: clip ±k·σ là no-op vì mu << σ ở tất cả k. Phân phối σ là return-scale (1.7–2.4%), mu là tiny. Cần chuẩn hoá lại hoặc dùng shrink factor.

2. **Sigma_shrink loại bỏ spike hoàn toàn** nhưng mất rel_score: k=1.5+ cho 0 spike, rel_score ~0.009. Đây là trade-off fundamental: shrink prediction về 0 → spike 0 nhưng mất signal.

3. **selective_strong (σ≤q40 AND |mu|≥q70)** rất hứa hẹn: DA 55.0%, ratio 0.253, spike 0.3 — nhưng coverage chỉ 2.4%. Không phải full-coverage solution.

4. **hetero_combined raw** vẫn là **best rel_score** (0.0372) — tốt hơn stressaux_w20 50%. Nhưng spike là vấn đề cần giải quyết ở lớp phía trên (selection/backtest), không phải shrinkage.

5. **Pattern chính**: σ head phân biệt được khi nào DA cao (55% ở selective_strong) — tín hiệu này nên được dùng trong **selection layer (P3)**, không phải là shrinkage thuần.

### Calibration σ (train-derived, apply on val)

| seed | sigma_med | sigma_q40 | sigma_q75 | sigma_q90 | mu_q70_abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 1.91% | 1.75% | 2.41% | 2.97% | 0.35% |
| 52 | 1.89% | 1.73% | 2.37% | 2.95% | 0.38% |
| 71 | 1.84% | 1.68% | 2.34% | 2.91% | 0.35% |

### Kết luận P2 và Quyết Định

**Primary finding**: **hetero_combined raw (rel_score 0.0372)** là candidate tốt nhất cho full-coverage. Spike cao hơn stressaux_w20 nhưng rel_score cao hơn 50%.

**Về spike control**: sigma_shrink không phải đáp án đúng vì sacrifice rel_score quá nhiều. Đáp án đúng là dùng σ làm **selection signal** (chọn stock nào, ngày nào để giao dịch) trong lớp filter phía sau.

**Quyết định tiếp theo**:
- ✅ Promote `hetero_combined` làm challenger candidate (rel_score 0.0372 > stressaux_w20 0.0248).
- ✅ P3: Build selection layer dùng `sigma` để filter ngày/mã spike — analogous với filter_signal.py nhưng dùng σ thay vì confidence proxy.
- ✅ Chạy 5-seed full run cho hetero_combined nếu P3 confirm improvement.
- ❌ Sigma_shrink full-coverage: loại bỏ (trade-off không có lợi).

---

## Step 7 (P3): σ-Based Selection Layer — ✅ Done

**Files**: `experiments/training/run_sigma_selection_probe.py`
**Artifacts**: `data/processed/assets/data_info_vn/history/training_runs/reports/sigma_selection_probe_20260521/`
**Gold**: `gold/vn_transition_pressure_20260512/plots/sigma_selection_probe_20260521/`

### Kết quả Aggregate (3 seeds, sorted by rel_score)

| Rule | rel_score | ± std | q90_max | spike≥8% | DA | ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q70 | **0.1288** | 0.0009 | 12.25% | 21.0 | 57.6% | 0.276 | 27% |
| conf_ratio_q60 | 0.1071 | 0.0105 | 11.47% | 18.7 | 55.9% | 0.249 | 37% |
| abs_mu_q70 | 0.0996 | 0.0037 | 12.40% | 19.3 | 56.7% | 0.245 | 34% |
| conf_ratio_q50 | 0.0864 | 0.0098 | 11.44% | 17.3 | 54.7% | 0.233 | 47% |
| abs_mu_q50 | 0.0788 | 0.0086 | 11.39% | 15.3 | 54.2% | 0.210 | 53% |
| **full_coverage** | 0.0372 | 0.0043 | 11.23% | 13.7 | 51.2% | 0.175 | 100% |
| daily_bottom_sigma_50pct | 0.0182 | 0.0042 | 8.88% | **2.0** | 50.3% | 0.172 | 50% |
| combo_s40_m70 | 0.0196 | 0.0144 | 7.65% | 0.3 | 55.0% | 0.253 | 2.4% |
| daily_bottom_sigma_25pct | 0.0160 | 0.0022 | 7.73% | 0.0 | 49.8% | 0.171 | 25% |

Ref: stressaux_w20 = rel_score 0.0248, spike 7.7, daily_q90_max 9.44%, coverage 100%.

### 4 Insight Quan Trọng

**1. Không có rule nào đồng thời beat stressaux_w20 trên CẢ rel_score VÀ spike**
- Muốn rel_score > 0.0248 → phải chấp nhận spike > 7.7 (và cao hơn nhiều).
- Muốn spike < 7.7 → phải sacrifice rel_score < 0.0248.
- Trade-off này là **fundamental** với model này — không phải bug.

**2. conf_ratio (= |mu|/sigma) là signal tốt nhất để select**
- conf_ratio_q70 (27% coverage): rel_score 0.1288, DA 57.6% — **signal chất lượng cao**.
- abs_mu_q70 cũng tương tự nhưng weaker (0.0996, DA 56.7%).
- Đây xác nhận σ head học được uncertainty calibration thực sự.

**3. hetero_combined = kiến trúc mạnh hơn stressaux_w20 về rel_score/signal chất**
- full_coverage: 0.0372 vs 0.0248 (+50%).
- Filtered @ 27% coverage: **0.1288** — gấp 5× production.
- DA filtered tăng từ 51.2% → 57.6% (signal trong tập chọn cao hơn nhiều).

**4. daily_bottom_sigma_50pct là tốt nhất nếu cần kiểm soát spike**
- rel_score 0.0182, spike 2.0, coverage 50%.
- Vẫn thấp hơn stressaux_w20 về rel_score nhưng spike giảm mạnh (2.0 vs 7.7).
- Dùng được như "low-risk filtered" version.

### Kết Luận Toàn Bộ Chuỗi P1→P2→P3

| Dimension | stressaux_w20 | hetero_combined |
| --- | --- | --- |
| Full-coverage rel_score | 0.0248 | **0.0372** (+50%) |
| Filtered rel_score (27% cov) | n/a | **0.1288** |
| Full-coverage spike≥8% | 7.7 | 13.7 (worse) |
| Low-spike filtered (spike ≤ 2) | via abstain (mất nhiều) | daily_bottom_sigma_50pct (0.0182, 50% cov) |
| DA (filtered) | ~50.7% | **57.6%** (at q70 conf_ratio) |

### Quyết Định Cuối

1. **Promote hetero_combined** làm candidate chính cho đường **filtered/selection** path:
   - Signal tốt hơn 50% trên full coverage.
   - Khi filter = conf_ratio_q70 (27% coverage): rel_score 0.1288, DA 57.6% → đủ tốt cho selection-based trading.

2. **Giữ stressaux_w20** làm anchor cho full-coverage production (spike thấp hơn).

3. **Bước tiếp theo**: wiring hetero_combined + conf_ratio filter vào selection/backtest pipeline:
   - Integrate với `src/models/selection/filter_signal.py` pattern.
   - Chạy rolling validation (w126/t21/s21) để đánh giá equity curve của conf_ratio_q70 selection.
   - Backtest với Wyckoff phase gate và transition_pressure rule.

4. **Chạy 5-seed full run** (60 epochs) cho hetero_combined nếu rolling validation pass.

---

## Step 8: Full 5-Seed Hetero Combined + Rolling Validation — ✅ Done

**Files**:
- `experiments/training/run_sigma_shrinkage_probe.py`
- `experiments/training/run_sigma_selection_probe.py`
- `experiments/training/run_hetero_rolling_backtest.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/sigma_selection_full5_20260521/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_rolling_backtest_20260521/`

**Gold mirrors**:
- `gold/vn_transition_pressure_20260512/plots/hetero_combined_full5_20260521/`
- `gold/vn_transition_pressure_20260512/plots/sigma_selection_full5_20260521/`
- `gold/vn_transition_pressure_20260512/plots/hetero_rolling_backtest_20260521/`

### 5-Seed Full Run (60 epochs, patience 15)

| Rule | rel_score | std | daily_q90_max | spike≥8% | DA | ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hetero_combined raw/full | **0.0339** | 0.0055 | 10.77% | 11.8 | 50.97% | 0.177 | 100% |
| stressaux_w20 reference | 0.0248 | 0.0072 | 9.44% | 7.7 | 50.7% | 0.166 | 100% |
| sigma_shrink_k1.5 | 0.0084 | 0.0019 | 7.70% | 0.0 | 50.97% | 0.082 | 100% |
| abstain_sigma_top10 | 0.0094 | 0.0022 | 8.17% | 1.0 | 49.95% | 0.162 | 79.7% |

**Conclusion**: 5-seed confirms hetero_combined is stronger on full-coverage rel_score (+0.0091 vs stressaux), but spike remains worse.

### 5-Seed Selection Readout

| Rule | rel_score | daily_q90_max | spike≥8% | DA | ratio | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q80 | **0.1293** | 15.18% | 22.8 | 59.05% | 0.313 | 19.2% |
| conf_ratio_q70 | 0.1222 | 11.54% | 20.2 | 56.93% | 0.269 | 28.5% |
| conf_ratio_q60 | 0.0991 | 11.04% | 17.0 | 55.42% | 0.246 | 38.2% |
| abs_mu_q50 | 0.0751 | 10.94% | 14.0 | 53.87% | 0.208 | 54.1% |
| full | 0.0339 | 10.77% | 11.8 | 50.97% | 0.177 | 100% |
| daily_bot_sig_50pct | 0.0158 | 8.62% | 1.8 | 50.13% | 0.174 | 50.3% |

**Conclusion**: fixed-split selection is very strong on rel_score and DA, but high-rel_score selections also increase spike. Low-sigma selections reduce spike but lose rel_score.

### Rolling Validation (w126/t21/s21, seed 43)

| Rule | Fold mean rel_score | Positive folds | DA | Coverage |
| --- | ---: | ---: | ---: | ---: |
| full | -0.0500 ± 0.0893 | 6/31 | 48.3% | 100% |
| conf_ratio_q70 | -0.1482 ± 0.2496 | 9/31 | 51.0% | 28.8% |
| daily_bot_sig_50pct | -0.0371 ± 0.0744 | 9/31 | 47.7% | 50.3% |

Equity backtest (long-only top-10, cost 15bps):

| Rule | Final equity | Ann ret | Sharpe | MaxDD | Turnover | Hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1.887 | 27.9% | 0.88 | -43.8% | 0.966 | 59.3% |
| conf_ratio_q70 | 1.846 | 26.8% | 0.90 | -41.0% | 1.011 | 59.4% |

### Interpretation

1. **Fixed-split hetero_combined passes rel_score improvement**: 0.0339 vs 0.0248.
2. **Rolling rel_score fails** because each rolling fold trains on only 126 days (~11k sequences), far below fixed-split train (~115k sequences). The model is undertrained/unstable on short windows.
3. **Equity remains positive despite negative rel_score**: ranking/trading signal survives even when absolute return calibration fails.
4. **Risk blocker**: turnover ~1/day and MaxDD > 40% are too high. This cannot be promoted until holding-period / turnover-constrained selector is added.

### Decision

- ✅ Keep `hetero_combined` as a **research challenger**, not production anchor.
- ✅ Next step should be **turnover-constrained holding-period selector**, not more loss/architecture tuning.
- ❌ Do not replace stressaux_w20 full-coverage anchor yet.
- ✅ Candidate direction: hetero_combined signal → holding-period selector → transition-pressure gate.

---

## Step 9: Holding-Period Grid + Transition Pressure Gate — ✅ Done

**Files**:
- `experiments/training/run_hetero_holding_period_grid.py`
- `experiments/training/run_hetero_pressure_gate_grid.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_holding_period_grid_20260521/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_pressure_gate_grid_20260521/`
- `gold/vn_transition_pressure_20260512/plots/hetero_holding_period_grid_20260521/`
- `gold/vn_transition_pressure_20260512/plots/hetero_pressure_gate_grid_20260521/`
- `gold/vn_transition_pressure_20260512/plots/hetero_pressure_gate_grid_20260521/equity_curves_top_candidates.png`

### Holding-Period Grid Results (5-seed ensemble, val 2020-04 → 2022-11)

Without pressure gate: raw hetero_combined at rebalance=1 loses money. Best Sharpe at rebalance=3.
Best low-turnover (TO<0.20): `full, reb=10, topk=8` → Sharpe 1.23, MaxDD -55%, Eq 2.60.
No combo passed: Sharpe>0.5 AND MaxDD>-25% AND TO<0.35 AND Eq>1.2.

### Transition Pressure Gate Results (pressure_delta_20 >= 0, active 62.2% of days)

Applying gate transforms the risk profile:

| Rule | Gate | Reb | TopK | Final Eq | Ann | Sharpe | MaxDD | Turnover | Worst Q |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q70 | nonneg | 10 | 8 | **2.676** | 45.7% | **2.064** | **-16.5%** | 9.6% | -1.83% |
| full | nonneg | 10 | 8 | **3.159** | 55.2% | 2.037 | -20.6% | 10.5% | -1.98% |
| conf_ratio_q70 | nonneg | 10 | 10 | 2.511 | 42.2% | 1.980 | -15.7% | 9.5% | -1.04% |
| conf_ratio_q80 | nonneg | 10 | 10 | 2.025 | 31.0% | 1.942 | -14.6% | 5.0% | -1.60% |
| daily_bot_sig | nonneg | 10 | 10 | 2.061 | 31.9% | 1.756 | -13.4% | 10.2% | -3.09% |
| daily_bot_sig | nonneg | 20 | 8 | 1.676 | 21.8% | 1.380 | **-11.4%** | 5.0% | -6.81% |

24 candidates pass production-like gate (Sharpe>0.5, MaxDD>-30%, TO<0.20, Equity>1.2).

### 5 Critical Insights

1. **Pressure gate is transformational**: Sharpe 0.88→2.06, MaxDD -44%→-16.5%. This is the single most important improvement.

2. **conf_ratio_q70 + pressure_nonneg + reb=10 + top=8** is the best balanced candidate:
   - Sharpe **2.064**, MaxDD **-16.5%**, Turnover 9.6%, worst quarter -1.83%
   - Beats production (Sharpe 0.59, MaxDD -24.3%) on all risk-adjusted metrics.

3. **full + pressure_nonneg + reb=10 + top=8** gives highest returns:
   - Final equity **3.159**, Ann return **55.2%**, Sharpe 2.04
   - Slightly worse MaxDD (-20.6%) than conf_ratio_q70

4. **conf_ratio_q80 + pressure_nonneg** is most conservative (TO=5%, positions=1.8-2.6)
   - Good risk control but fewer positions = more concentration risk

5. **Validation discipline matters**: All metrics are val-only (2020-04→2022-11). Holdout still closed.

### Decision

✅ **Promote `conf_ratio_q70 + pressure_nonneg + reb=10 + topk=8`** as primary candidate:
- Research-grade, not production-ready (validation-derived, holdout not opened).
- Next step: multi-seed rolling validation with this exact config to confirm stability.

✅ **Full + pressure_nonneg + reb=10 + topk=8** as secondary (higher returns, slightly more risk).

⚠️ Position count (3.7–4.6) is below minimum 6 required by current policy.
Need to explore topk=15–20 at same rule to meet min_positions constraint.

### Summary of Entire Research Chain (Steps 5–9)

| Approach | rel_score | Sharpe (portfolio) | MaxDD | Turnover |
| --- | ---: | ---: | ---: | ---: |
| stressaux_w20 (production) | 0.0248 | ~0.59 | -24.3% | 0.179 |
| hetero_combined full | 0.0339 | 0.88 | -43.8% | 0.97 |
| + holding-period (reb=10) | n/a | 1.23 | -55% | 0.17 |
| + pressure_nonneg gate | n/a | **2.064** | **-16.5%** | 0.096 |

**The combination of hetero_combined σ-signal + conf_ratio selection + pressure gate is substantially better than production on all portfolio metrics, while requiring holdout validation to confirm.**

---

## Step 10: Pressure-Gated TopK 3–5 Concentrated Probe — ✅ Done

**Files**: inline probe using existing `run_hetero_holding_period_grid.py` and `run_hetero_pressure_gate_grid.py` helpers.

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_pressure_gate_top3_5_20260521/`
- `gold/vn_transition_pressure_20260512/plots/hetero_pressure_gate_top3_5_20260521/`

### Top Results

| Rule | Reb | TopK | Final Eq | Ann | Sharpe | MaxDD | Turnover | Mean Pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q80 | 20 | 5 | 2.802 | 48.3% | 1.918 | -13.8% | 3.9% | 1.46 |
| full | 10 | 5 | 3.175 | 55.6% | 1.914 | -23.4% | 10.7% | 2.88 |
| conf_ratio_q80 | 20 | 4 | 2.711 | 46.4% | 1.858 | -16.6% | 4.0% | 1.24 |
| conf_ratio_q80 | 10 | 5 | 2.898 | 50.2% | 1.843 | -17.6% | 9.8% | 1.82 |
| daily_bot_sig_50pct | 10 | 5 | 1.897 | 27.7% | 1.511 | -12.3% | 11.1% | 2.88 |

### Interpretation

- TopK 3–5 is viable as a concentrated research sleeve.
- Best risk-adjusted candidate: `conf_ratio_q80 + pressure_nonneg + rebalance=20 + top_k=5`.
- It improves MaxDD (-13.8%) and turnover (3.9%) versus top_k=8 candidate, but mean positions is only 1.46 because the pressure/confidence filter is sparse.
- Not advisor-safe if current policy requires min_positions ≥6.

### Decision

✅ Keep `conf_ratio_q80 + pressure_nonneg + rebalance=20 + top_k=5` as **concentrated sleeve candidate**.
⚠️ Keep `conf_ratio_q70 + pressure_nonneg + rebalance=10 + top_k=8/10` as advisor-track candidate because it has more positions.

---

## Step 11: Short-Term Stop-Cash Overlay — ✅ Done

**Files**:
- `experiments/training/evaluate_stop_cash_overlay.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_stop_cash_overlay_20260524/`
- `gold/vn_transition_pressure_20260512/plots/hetero_stop_cash_overlay_20260524/`

### Scope

Offline overlay on existing rolling daily returns. No retraining. Holdout/test not used.
Rule: if trailing portfolio return over lookback window breaches a loss threshold, go cash for cooldown days.

### Best result

| Policy | Rule | Worst Eq before | Worst Eq after | Mean Eq before | Mean Eq after | Mean Sharpe before | Mean Sharpe after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q70 r20 k20 m6 | lookback=14, threshold=10%, cooldown=10 | 0.8243 | **0.8467** | 1.0172 | 1.0169 | 1.7543 | 1.7037 |
| daily_bot_sig r20 k20 m5 | lookback=14, threshold=10%, cooldown=10 | 0.8232 | **0.8465** | 1.0173 | 1.0178 | 1.9737 | 1.9827 |
| abs_mu_q60 r20 k20 m6 | lookback=14, threshold=10%, cooldown=10 | 0.7947 | **0.8348** | 1.0238 | 1.0232 | 2.3443 | 2.2689 |

### Interpretation

- Stop-cash helps worst-fold equity by ~2–4 percentage points.
- It does not solve the main issue: worst MaxDD remains around -25% to -27% because the trigger fires after the crash starts.
- Best short-term candidate remains `daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5` with stop-cash lookback=14, threshold=10%, cooldown=10.
- Next improvement should use a **pre-crash market stress gate**, not portfolio-loss stop after-the-fact.

### Decision

✅ Keep stop-cash overlay as small risk reducer.
⚠️ Do not rely on it as the primary crash protection.
➡️ Next: test market stress gate using market drawdown / volatility / breadth before fold-26 style crashes.

---

## Step 12: Wyckoff Strict Gate (Pre-Crash Signal) — ✅ Done

**Files**: inline analysis using existing `run_hetero_long_finetune_batch.py` artifacts.

**Gate definition**:
- `wyckoff_strict = (pressure_delta_20 >= 0) AND (wyckoff_phase_60d >= 0.35)`
- `wyckoff_phase_60d` is already in the VN gold feature set.

### Result vs original pressure_only gate

| Policy | Original worst fold | Wyckoff strict worst fold | Original Sharpe | Wyckoff Sharpe | Active % |
| --- | ---: | ---: | ---: | ---: | ---: |
| conf_ratio_q70 r20 k20 m5 | 0.8243 | **0.9117** | 1.008 | 1.064 | 61.8% |
| conf_ratio_q70 r20 k20 m6 | 0.8243 | **0.9117** | 0.987 | 1.049 | 61.8% |
| daily_bot_sig r20 k20 m5 | 0.8232 | **0.9089** | 1.153 | 1.144 | 61.8% |
| abs_mu_q60 r20 k20 m6 | 0.7947 | 0.8834 | 1.195 | 1.184 | 61.8% |

### Why it works

- Fold 26 (Jun 2022 crash): `wyckoff_strict` active only **14.3%** vs `pressure_only` 38.1%.
- `wyckoff_phase_60d` was already at 0.27 before the crash — a pre-crisis signal, not lagging.
- Gate reduces active days from 62.2% → ~61.8% globally (only blocks bad periods).

### Trade-off

- Worst fold improves: 0.82 → 0.91 (+9pp on worst case)
- Cumulative return decreases: e.g., 20.7% → 17.7% annualized (daily_bot_sig, seed 43)
- Mean Sharpe broadly maintained or slightly improved

### Decision

✅ **Replace `pressure_only` with `wyckoff_strict`** as the primary market gate for all hetero_combined candidates.
✅ Best advisor-track candidate: `conf_ratio_q70_pressure_nonneg_r20_k20_m5` with `wyckoff_strict` gate:
- Worst fold equity: **0.9117** (up from 0.8243)
- Mean Sharpe: **1.064** (up from 1.008)
- Annual return: ~18–19%
- Active: 61.8% of days

⚠️ Still no rolling candidate passes strict worst_fold > 1.0 gate — June 2022 crash is a 9% loss even with wyckoff filter active. This is a systematic market risk.

### Next valid step

- Wire `wyckoff_strict` gate into `src/models/selection/filter_signal.py` and rerun strict rolling validation.
- Run seeds 71, 82 to confirm 5-seed stability.
- After confirmation → one-time holdout read.

---

## Step 13: Wyckoff Strict Metric Series (rel_score + abs(E)) — ✅ Done

**Files**:
- `experiments/training/evaluate_wyckoff_strict_metric_series.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_wyckoff_metric_series_20260524/`
- `gold/vn_transition_pressure_20260512/plots/hetero_wyckoff_metric_series_20260524/`

### Policy comparison (93 fold-seeds, rolling, validation only)

| Policy | mean_rel_score | positive_rel_folds | mean_absE_robust | mean_absE_q90 | mean_DA | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **daily_bot_sig wyckoff_strict r20 k20 m5** | **-0.0862** | 24/93 | **3.10%** | **3.75%** | 49.4% | 13.5% |
| abs_mu_q60 wyckoff_strict r20 k20 m6 | -0.1228 | 23/93 | 4.57% | 5.27% | 50.1% | 11.8% |
| conf_ratio_q70 wyckoff_strict r20 k20 m5 | -0.1648 | 24/93 | 4.18% | 4.88% | 51.1% | 9.2% |
| conf_ratio_q70 wyckoff_strict r20 k20 m6 | -0.1695 | 24/93 | 4.17% | 4.87% | 51.0% | 9.1% |

*Benchmark (stressaux_w20 fixed-split full-universe): rel_score +0.025, absE_robust ≈ 3.4%, absE_q90 ≈ 4.86%, DA 50.7%*

### Key findings

1. **rel_score âm trong rolling**: do mỗi fold chỉ train 126 ngày — model underfit, calibration yếu. Đây không phải lỗi của selection rule, mà là giới hạn của rolling protocol với short train window.

2. **daily_bot_sig có abs(E) tốt nhất**:
   - absE_robust trung bình = **3.10%** — gần tương đương production (3.4%)
   - absE_q90 median = **2.87%** — thấp hơn production (4.86%)
   - Days với absE_q90 > 3.5%: chỉ 29.6% (vs production ~45%)
   - Days với absE_q90 > 5.0%: chỉ **9.0%**

3. **conf_ratio_q70 có DA cao nhất (51.1%)** nhưng abs(E) cao hơn — chọn stocks mạnh hơn nhưng error lớn hơn.

4. **Trend cải thiện theo năm** cho conf_ratio_q70:
   - 2020: rel_score -0.281, DA 48.9% (model còn underfit)
   - 2021: rel_score -0.129, DA 52.5%
   - 2022: rel_score -0.042, DA 51.8%

### Interpretation

**daily_bot_sig_50pct + wyckoff_strict** là candidate tốt nhất khi nhìn từ abs(E):
- abs(E) thấp nhất
- Worst fold equity tốt nhất (0.9089)
- Số ngày spike > 3.5% thấp nhất

**conf_ratio_q70 + wyckoff_strict** là candidate tốt nhất khi nhìn từ portfolio perspective:
- DA cao nhất (51.1%)
- Worst fold equity 0.9117
- Nhưng abs(E) cao hơn daily_bot_sig

### Decision

✅ Promote **`daily_bot_sig_50pct_wyckoff_strict_r20_k20_m5`** làm **primary candidate** cho abs(E)/prediction quality focus.
✅ Promote **`conf_ratio_q70_wyckoff_strict_r20_k20_m5`** làm **secondary candidate** cho portfolio/DA focus.
✅ Next valid step: run seeds 71,82 để 5-seed confirmation → holdout read.

---

## Step 14: 5-Seed Confirmation + Gate Tightening — ✅ Done

**Files**:
- `experiments/training/run_hetero_long_finetune_batch.py`
- `experiments/training/evaluate_market_gate_overlays.py`
- `experiments/training/evaluate_wyckoff_strict_metric_series.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_long_finetune_batch_20260522/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_market_gate_overlays_20260524/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_wyckoff_metric_series_5seed_20260524/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/hetero_gate_metric_compare_20260524/`

### Scope

Validation-only. Holdout/test not used.
Added seeds 71 and 82, then recomputed all cached rolling policies for seeds 43,52,62,71,82.

### Best gate overlay (5 seeds)

Best portfolio-risk candidate is `daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5 + wyck040`:

| Metric | Value |
| --- | ---: |
| gate active | 59.5% |
| mean equity | 1.523 |
| min seed equity | 1.442 |
| mean annualized return | 17.7% |
| mean Sharpe | 1.244 |
| worst max DD | -18.8% |
| worst fold equity | **0.9532** |
| positive folds | 77/155 |

This is materially better than prior `wyck035` on worst fold (0.9089 → 0.9532), while preserving Sharpe > 1.

### abs(E)/rel_score comparison for daily_bot_sig across gates

| Gate | mean_rel_score | mean_absE_robust | mean_absE_q90 | p90_absE_robust | DA | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wyck040 | -0.1085 | **3.04%** | **3.69%** | **4.18%** | 48.3% | 13.2% |
| wyck035_idx10 | -0.0997 | 3.10% | 3.74% | 4.22% | 48.4% | 13.4% |
| wyck035 | -0.0940 | 3.12% | 3.77% | 4.23% | 48.6% | 13.5% |

### Interpretation

- `wyck040` is the best current risk/error-control gate.
- It slightly worsens rel_score versus `wyck035` because it trades less, but improves abs(E), q90 error, and worst-fold equity.
- Full rolling rel_score remains negative due to short 126-day rolling train windows, but abs(E) is now competitive with production while worst crash risk is much lower.

### Decision

✅ Promote `daily_bot_sig_50pct + wyck040 + rebalance=20 + top_k=20 + min_positions=5` as the current validation-only primary candidate.
✅ Keep `conf_ratio_q70 + wyck035/ultra_safe` as secondary portfolio/DA sidecar only.
❌ Do not open holdout yet.

### Next step

Wire `wyck040` into the selection pipeline as a named gate, then produce a clean advisor-style report with:
- equity curve
- fold equity distribution
- daily abs(E) q90 series
- rel_score/abs(E) year-month tables

---

## Step 15: Fixed Long-Train RelScore Calibration — ✅ Done

**File**:
- `experiments/training/evaluate_fixed_train_relscore_calibration.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_relscore_calibration_20260524/`
- `gold/vn_transition_pressure_20260512/plots/fixed_train_relscore_calibration_20260524/`

### Protocol

This matches the intended setup:
- Train: all data `<= 2020-03-31`
- Validation/in-sample: `2020-04-01 -> 2022-11-15`
- Holdout/test: not used

Uses saved 5-seed full-train predictions from `hetero_combined_full5_20260521`.

### Key result

| Variant | Gate | rel_score | absE_robust | absE_q90 | DA | pred/actual q90 ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train_cal | none | **0.03493** | **3.637%** | 4.761% | 50.97% | 0.184 |
| raw | none | 0.03395 | 3.641% | 4.769% | 50.97% | 0.177 |
| clip_vol_1.0 | none | 0.03401 | 3.641% | 4.768% | 50.97% | 0.177 |
| train_cal | pressure_only | 0.00935 | 3.734% | 4.959% | n/a | 0.105 |
| train_cal | wyck035 | 0.00765 | 3.740% | 4.968% | n/a | 0.101 |
| train_cal | wyck040 | 0.00669 | 3.744% | 4.973% | n/a | 0.099 |

### Interpretation

- The previous negative rolling rel_score came from the short rolling train window (`w126`), not from the model itself.
- With the correct long-train setup, hetero_combined is **positive rel_score ~0.035**, beating stressaux_w20 reference ~0.0248.
- Train-fold calibration learns scale around `1.04`, giving a small gain over raw.
- Market gates improve portfolio risk but reduce full-universe rel_score because inactive days are converted to zero predictions. Do not judge portfolio gates by full-universe rel_score alone.

### Decision

✅ For rel_score target, use **long-train/fixed-train hetero_combined + train_cal** as primary prediction candidate.
✅ For portfolio/risk target, use **daily_bot_sig + wyck040** as execution overlay.
❌ Do not use `w126` rolling rel_score as the main model-quality metric.

### Next step

Run long-window rolling only as robustness stress-test with `w504/w756/expanding`, not `w126`, then freeze before holdout.

---

## Step 16: Fixed Long-Train Fold Series + Seed Ensemble — ✅ Done

**Files**:
- `experiments/training/evaluate_fixed_train_ensemble_calibration.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_fold_relscore_series_20260524/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_ensemble_calibration_20260524/`
- `gold/vn_transition_pressure_20260512/plots/fixed_train_fold_relscore_series_20260524/`
- `gold/vn_transition_pressure_20260512/plots/fixed_train_ensemble_calibration_20260524/`

### Protocol

- Train: all data `<= 2020-03-31`
- Validation/in-sample: `2020-04-01 -> 2022-11-15`
- Holdout/test: not used
- Prediction source: cached `hetero_combined_full5_20260521`

### Fixed long-train fold series

This slices the validation period into 21-day folds without retraining on short rolling windows.

| Variant | mean fold rel_score | median fold rel_score | positive folds | absE_robust | absE_q90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_cal_1_04 | **0.0232** | 0.0157 | 109/155 | **3.551%** | **4.507%** |
| raw | 0.0232 | 0.0165 | 108/155 | 3.552% | 4.511% |
| shrink_075 | 0.0221 | 0.0156 | 112/155 | 3.559% | 4.533% |
| zero | 0.0000 | 0.0000 | 0/155 | 3.662% | 4.709% |

Year split for `train_cal_1_04`:

| Year | mean fold rel_score | positive folds | mean absE_robust |
| --- | ---: | ---: | ---: |
| 2020 | 0.0178 | 32/50 | 3.119% |
| 2021 | 0.0237 | 47/60 | 3.486% |
| 2022 | 0.0286 | 30/45 | 4.119% |

Daily rel_score remains weakest in 2022, especially crash/dislocation days, but fold-level fixed-train rel_score stays positive across all years.

### Seed ensemble result

Best validation-only prediction candidate is now the calibrated mean ensemble:

| Variant | rel_score | absE_robust | absE_q90 | DA | pred/actual q90 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ensemble_mean_cal_each_traincal_clip | **0.04478** | **3.600%** | **4.693%** | **51.83%** | 0.193 |
| ensemble_mean_cal_each_traincal | 0.04476 | 3.600% | 4.695% | 51.83% | 0.193 |
| ensemble_mean_raw_traincal | 0.04391 | 3.604% | 4.704% | 51.79% | 0.186 |
| single-seed train_cal reference | 0.03493 | 3.637% | 4.761% | 50.97% | 0.184 |

The ensemble improves both target quality and stability:
- rel_score improves from `0.03493` to `0.04478`
- DA improves from `50.97%` to `51.83%`
- absE_robust improves from `3.637%` to `3.600%`
- absE_q90 improves from `4.761%` to `4.693%`

### Additional checks

- Regime shrink overlays using pressure/Wyckoff/index filters did **not** improve full-universe rel_score.
- Keep market gates for portfolio execution only; do not gate full-universe prediction metrics.
- Rolling retrain `w504` remains a stress-test and is still not the target metric.

### Decision

✅ Promote `hetero_combined_full5 -> mean ensemble -> train calibration` as the current prediction/rel_score candidate.
✅ Keep `daily_bot_sig_50pct + wyck040 + rebalance=20 + top_k=20 + min_positions=5` as the portfolio/risk overlay.
❌ Holdout remains closed.

---

## Step 17: Frozen Validation Candidate Advisor Report — ✅ Done

**File**:
- `experiments/training/build_frozen_validation_candidate_report.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/frozen_validation_candidate_20260524/`
- `gold/vn_transition_pressure_20260512/plots/frozen_validation_candidate_20260524/`

### Protocol

No new training. Composed from cached artifacts:
- Prediction metrics: `fixed_train_ensemble_calibration_20260524` (5 seeds, ensemble mean, train calibration)
- Portfolio metrics: `hetero_market_gate_overlays_20260524` (5 seeds, daily_bot_sig + wyck040)

### Frozen candidate decision

#### Prediction candidate
- **Method**: `hetero_combined_full5` → mean-ensemble (5 seeds) → per-seed train calibration scale
- **Variant**: `ensemble_mean_cal_each_traincal_clip`

| Metric | Frozen ensemble | Prior single-seed train_cal |
| --- | ---: | ---: |
| rel_score | **0.04478** | 0.03493 |
| absE_robust | **3.60%** | 3.64% |
| absE_q90 | **4.69%** | 4.76% |
| DA | **51.83%** | 50.97% |
| pred/actual q90 ratio | **0.193** | 0.184 |

#### 21-day fold stability
| Metric | Value |
| --- | ---: |
| mean fold rel_score | 0.02980 |
| positive folds | 24/32 |
| min fold rel_score | -0.02823 |
| mean absE_robust | 3.64% |

#### Yearly series (daily level)
| Year | mean rel_score | positive days | mean absE_robust |
| --- | ---: | ---: | ---: |
| 2020 | 0.01254 | 103/193 | 2.915% |
| 2021 | 0.01448 | 143/250 | 3.348% |
| 2022 | 0.00586 | 116/216 | 4.198% |

> 2022 has higher absE due to higher volatility; rel_score still positive.

#### Portfolio overlay candidate
- **Policy**: `daily_bot_sig_50pct_pressure_nonneg_r20_k20_m5` + gate `wyck040`
- Gate active: 59.48% of days

| Metric | Value |
| --- | ---: |
| mean equity | 1.5231 |
| min seed equity | 1.4417 |
| mean annual return | 17.67% |
| mean Sharpe | 1.2436 |
| worst max drawdown | -18.75% |
| worst fold equity | 0.9532 |
| positive folds | 77/155 |

### Decision

✅ Frozen prediction candidate: `ensemble_mean_cal_each_traincal_clip` from `hetero_combined_full5_20260521`.
✅ Frozen portfolio overlay: `daily_bot_sig_50pct + wyck040 + rebalance=20 + top_k=20 + min_positions=5`.
✅ Both candidates documented with full series (yearly, monthly, fold-level, worst-case).
❌ Holdout remains closed until user explicitly approves.

### Next step

If user is satisfied with this frozen candidate, the next step is to open holdout once (final evaluation only, no further tuning).

---

## Step 18: Academic Ablation + Baseline + Significance — ✅ Done

**File**:
- `experiments/training/run_academic_ablation_baseline_significance.py`

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/academic_ablation_baseline_significance_20260525/`
- `gold/vn_transition_pressure_20260512/plots/academic_ablation_baseline_significance_20260525/`

### Protocol

- Train: all data `<= 2020-03-31`
- Validation/in-sample: `2020-04-01 -> 2022-11-15`
- Holdout/test: not used
- Fold significance: paired 21-day validation folds, bootstrap resampling (`n=20,000`)

### Baselines and ablations

| Model | rel_score | absE_robust | absE_q90 | DA |
| --- | ---: | ---: | ---: | ---: |
| ensemble_mean_cal_each_train_cal_clip | **0.04478** | **3.600%** | **4.693%** | **51.83%** |
| ensemble_mean_cal_each_train_cal | 0.04476 | 3.600% | 4.695% | 51.83% |
| ensemble_mean_raw_train_cal | 0.04391 | 3.604% | 4.704% | 51.79% |
| ensemble_mean_train_cal_each | 0.03922 | 3.621% | 4.748% | 51.83% |
| ensemble_median_train_cal | 0.03860 | 3.624% | 4.748% | 51.75% |
| single_seed43_train_cal | 0.03597 | 3.634% | 4.770% | 51.21% |
| single_seed43_raw | 0.03498 | 3.637% | 4.776% | 51.21% |
| global_train_mean | 0.00054 | 3.767% | 5.011% | 47.46% |
| zero | 0.00000 | 3.769% | 5.022% | 7.46% |
| stock_train_mean | -0.00116 | 3.773% | 5.018% | 46.81% |
| ridge_last_step | -0.00313 | 3.781% | 4.997% | 48.26% |
| lagged_stock_mean5_val_only | -0.07295 | 4.044% | 5.034% | 46.16% |

### Paired bootstrap significance

Candidate: `ensemble_mean_cal_each_train_cal_clip`.

| Baseline | mean Δ rel_score | 95% CI | p(Δ <= 0) | Positive folds |
| --- | ---: | ---: | ---: | ---: |
| zero | +0.02980 | [0.01650, 0.04409] | 0.0000 | 24/32 |
| global_train_mean | +0.03048 | [0.01712, 0.04497] | 0.0000 | 25/32 |
| stock_train_mean | +0.03048 | [0.01731, 0.04428] | 0.0000 | 25/32 |
| ridge_last_step | +0.02956 | [0.01722, 0.04293] | 0.0000 | 23/32 |
| single_seed43_train_cal | +0.01234 | [0.00423, 0.02436] | 0.0001 | 23/32 |
| ensemble_median_train_cal | +0.00190 | [-0.00081, 0.00472] | 0.0866 | 18/32 |
| ensemble_mean_train_cal_each | +0.00026 | [-0.00181, 0.00250] | 0.4207 | 12/32 |
| ensemble_mean_raw_train_cal | -0.00008 | [-0.00095, 0.00080] | 0.5672 | 16/32 |

### Academic interpretation

- Strong paper-grade claim: **calibrated mean ensemble significantly outperforms zero, mean, ridge, lagged, and single-seed baselines** on paired validation folds.
- Weak/no claim: the final `clip` variant is **not significantly different** from the simpler calibrated mean ensemble variants.
- For the paper, describe the method as **RelScore-calibrated heteroscedastic seed ensemble**, not as a sigma-clip invention.
- The main methodological novelty should be framed as:
  1. robust rel_score-first forecasting objective/evaluation,
  2. heteroscedastic return forecasting with seed ensembling,
  3. train-only calibration,
  4. separation between prediction quality and execution risk overlay.

### Multi-market next step

VN evidence is now academically stronger. To make the paper more ambitious, next run should repeat the same protocol on:
- `data/processed/assets/data_info_us/history/us_gold_recommended.csv`
- `data/processed/assets/data_info_jp/history/jp_gold_recommended.csv`

The target is not necessarily identical dates, but identical **walk-forward principle**:
- choose a pre-event train cutoff per market,
- keep validation closed for tuning,
- reserve holdout as final one-shot confirmation,
- report the same ablation, baseline, and bootstrap tables.

---

## Step 19: Multi-Market Portable Baseline + Hetero Smoke — ✅ Done

**Files**:
- `experiments/training/run_multimarket_portable_baseline_significance.py`
- `experiments/training/run_hetero_nll_probe.py` (added `--feature-columns` and `--variants`)

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_baseline_significance_20260525/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_hetero_summary_20260525/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/portable_hetero_3seed_20260525_vn/`
- `data/processed/assets/data_info_us/history/training_runs/reports/portable_hetero_3seed_20260525_us/`
- `data/processed/assets/data_info_jp/history/training_runs/reports/portable_hetero_3seed_20260525_jp/`
- gold mirrors under `gold/vn_transition_pressure_20260512/plots/`

### Protocol

- Markets: VN, US, JP
- Common portable features only: OHLCV/technical features shared by all three datasets
- Train: `<= 2020-03-31`
- Validation/in-sample: `2020-04-01 -> 2022-11-15`
- Holdout/test: not used
- Model smoke: `hetero_combined`, 3 seeds (`43,52,62`), epochs=18, patience=5

### Simple portable baseline result

Simple baselines are near zero across all markets:
- VN: best simple baseline ≈ `0.0000` rel_score (`zero`)
- US: best simple baseline ≈ `0.00235` rel_score (`global_train_mean`, drift baseline)
- JP: best simple baseline ≈ `0.00068` rel_score (`ridge_portable`)

### Portable hetero 3-seed result

| Market | rel_score mean | rel_score std | DA mean | pred/actual q90 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| VN | 0.00256 | 0.00557 | 47.65% | 0.090 | positive, but far below VN-specific `0.04478` |
| US | 0.00229 | 0.00104 | 51.38% | 0.098 | roughly tied with drift baseline |
| JP | 0.00102 | 0.00290 | 51.26% | 0.085 | slightly above best simple baseline |

### Interpretation

- The portable core is mildly positive in all three markets, but not yet strong enough for a final multi-market paper claim.
- VN-specific context features and calibration are clearly important: VN portable `0.00256` vs VN-specific frozen candidate `0.04478`.
- Multi-market narrative should be framed as:
  - **portable robust forecasting core** generalizes weakly across markets,
  - **market-specific context/gates/calibration** create strong local performance,
  - final paper needs full 5-seed calibrated ensemble and drift/market-demeaned significance tests for US/JP.

### Next step

Before opening any holdout, run a full academic multi-market package:
1. Save US/JP predictions for all seeds.
2. Build train-only calibrated mean ensemble per market.
3. Add market-demeaned / cross-sectional alpha metric to avoid drift-only wins.
4. Bootstrap paired fold significance against zero, ridge, global mean, and drift-controlled baselines.

---

## Step 20: Multi-Market Portable Ensemble + Alpha Evaluation — ✅ Done

**Files**:
- `experiments/training/evaluate_multimarket_portable_ensemble.py`
- `experiments/training/run_hetero_nll_probe.py` (now saves train/validation predictions with `--save-predictions`)

**Artifacts**:
- `data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_ensemble_academic_20260525/`
- `gold/vn_transition_pressure_20260512/plots/multimarket_portable_ensemble_academic_20260525/`
- `data/processed/assets/data_info_vn/history/training_runs/reports/portable_hetero_5seed_preds_20260525_vn/`
- `data/processed/assets/data_info_us/history/training_runs/reports/portable_hetero_5seed_preds_20260525_us/`
- `data/processed/assets/data_info_jp/history/training_runs/reports/portable_hetero_5seed_preds_20260525_jp/`

### Protocol

- Markets: VN, US, JP
- Features: common portable OHLCV/technical set only
- Train: `<= 2020-03-31`
- Validation/in-sample: `2020-04-01 -> 2022-11-15`
- Seeds: `43,52,62,71,82`
- Model: `hetero_combined`
- Ensemble selection: train-only selection among raw/calibrated mean/median ensembles
- Added strict `alpha_rel_score`: date-wise cross-sectional demeaned returns/predictions
- Holdout/test: not used

### Main results

| Market | Selected portable rel_score | Selected alpha_rel_score | Fold mean rel_score | Positive folds | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| VN | 0.00613 | 0.00285 | 0.00539 | 23/32 | portable positive; VN-specific candidate still much stronger (`0.04478`) |
| US | ~0.003 (selected train-only; best raw mean 0.00604) | negative | -0.00164 | 16/32 | regular rel_score can be drift-like; alpha weak |
| JP | ~0.0008 selected; best median raw 0.00450 | slightly negative | ~0.00008 | 15/31 | weak/no robust significance |

### Significance read

- VN portable selected candidate beats zero/global/ridge on paired fold rel_score at about p≈0.03–0.04, but margin is small and alpha effect is weak.
- US/JP portable candidates do **not** show robust paired-fold significance after train-only selection.
- Alpha/demeaned metric is the important stress test: it prevents interpreting drift-only improvement as stock-selection skill.

### Paper implication

This improves the paper narrative but also sets the boundary honestly:

✅ Strong claim remains VN-specific: `RelScore-calibrated heteroscedastic seed ensemble + VN context`.
✅ Portable core shows weak cross-market signal and is useful as a framework component.
❌ Not yet enough to claim the same model works strongly across US/JP.

Recommended paper framing:
- Main paper: VN/emerging-market robust forecasting with ablation/significance.
- Multi-market section: portability stress test; show that common OHLCV core is mildly positive but market-specific context is necessary.
- Future work: market-specific context adapters for US/JP, not a universal single model claim.

---

## Step 21: Framework Branch + Context Adapter Probe — ✅ Done

**Branch**:
- `codex/multimarket-framework`

**Files**:
- `docs/multimarket_framework_branch_manifest_20260525.md`
- `experiments/training/run_multimarket_context_adapter_probe.py`

**Cleanup**:
- Removed tracked `.DS_Store` files from the branch index.
- Branch policy: commit code + compact summaries only; keep heavy `.npz`/prediction folders local unless explicitly promoted.

### Context adapter protocol

- Markets: VN, US, JP
- Train: `<= 2020-03-31`
- Validation: `2020-04-01 -> 2022-11-15`
- Holdout/test: not used
- Lightweight model: Ridge only, as pre-LSTM adapter screening
- Portable features: common OHLCV/technical set
- Context adapter features:
  - market return 1/5/20
  - market volatility 20
  - market breadth positive ratio
  - market absolute return
  - cross-sectional ranks for momentum, volatility, MA gap, volume change

### Result

| Market | Δ rel_score adapter vs portable | Δ alpha_rel_score adapter vs portable | Interpretation |
| --- | ---: | ---: | --- |
| VN | +0.00037 | +0.00038 | neutral/small |
| US | -0.00358 | **+0.00348**, p≈0.0076 | adapter improves stock-selection alpha but hurts drift/raw rel_score |
| JP | -0.00127 | +0.00052 | weak/no significant gain |

### Interpretation

- The adapter is useful for the **framework narrative** because it separates market drift from stock-selection alpha.
- US result is especially informative: context improves alpha significantly but raw rel_score worsens, meaning raw metric can hide stock-selection effects.
- Next LSTM framework should train/evaluate two heads or two objectives:
  1. raw return rel_score for prediction quality,
  2. date-demeaned alpha rel_score for stock selection.

### Next step

Build a market-context adapter LSTM experiment:
- use portable core + context features,
- report both raw rel_score and alpha_rel_score,
- do not replace VN-specific anchor unless it beats `0.04478`,
- keep holdout closed.
