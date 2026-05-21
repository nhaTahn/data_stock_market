# Deferred LSTM Improvements — 2026-05-14

Hai cải tiến từ Phase L của plan đã được **đánh giá và hoãn lại** vì
risk/reward không thuận lợi với scope hiện tại (VN30 ~29 mã, window=15).
Ghi nhận để không quên — nếu mở rộng universe hoặc đổi setup, xem lại.

## L2 — Per-stock embedding (replace one-hot)

**Trạng thái hiện tại:**

- `use_stock_identity` đã được implement nhưng dưới dạng
  [one-hot encoding](../src/models/training/pipeline.py:731-748) appended vào
  mỗi timestep của input sequence (29-dim cho VN30, repeat × 15 step).
- Plan ban đầu (L2) đề xuất thay one-hot bằng `keras.layers.Embedding(num_stocks, dim=8)`.

**Lý do hoãn:**

- Universe quá nhỏ (~29 stocks). Embedding(29 → 8) ≈ linear projection của
  one-hot — về mặt biểu diễn không khác nhau đáng kể, model sẽ học gần như
  cùng một mapping.
- Lợi ích thực sự của embedding (chia sẻ thông tin qua similarity giữa
  stocks) chỉ thể hiện khi `num_stocks ≥ vài trăm` và có cấu trúc cluster
  rõ (sectors lớn, market caps).
- Effort: cần sửa 5-6 file (`embeddings.py` mới, signmag dual-input,
  datasets stock_id meta, fitters dual feed, pipeline assembly, model
  save/load). Rủi ro break model load/save không tương xứng với expected
  gain `≈ 0` đến `+0.0005` rel_score.

**Điều kiện để revisit:**

- Khi mở rộng training sang multi-market portable (VN + US + JP + HK + KR)
  với `num_stocks ≥ 200`.
- Khi có **sector embedding** muốn share parameters across stocks trong
  cùng sector (lúc đó embedding rất có ý nghĩa, không phải một-hot rời rạc).
- Khi muốn export model làm transfer learning sang market mới.

**Đường đi đề xuất khi quay lại:**

1. Tạo `src/models/architectures/embeddings.py` với `build_stock_embedding(num_stocks, embed_dim)`.
2. Modify `signmag.py` để nhận dual input `[x_seq, stock_id]`; concat embedding với LSTM output trước Dense heads.
3. Modify `datasets.py` export `stock_id` (int) trong meta.
4. Modify `fitters.py` để feed `[x, stock_id]`.
5. Modify `pipeline.py` để build stock_to_idx (đã có) → stock_id arrays cho train/val/test.
6. A/B trên WF-CV: one-hot (current) vs embedding(dim=4/8/16).

## L3 — Attention pooling on LSTM hidden states

**Trạng thái hiện tại:**

- Backbone `build_lstm_backbone` ([backbone.py](../src/models/architectures/backbone.py))
  trả về **last-step hidden state** của LSTM (`return_sequences=False` ở
  layer cuối) — model coi mỗi timestep trong window=15 có "memory" như
  nhau.
- Có family `attention` riêng ([attention.py](../src/models/architectures/attention.py))
  dùng `MultiHeadAttention` nhưng:
  - Docs `current_best_path.md` ghi: "experimental, không beat plain LSTM trong thử nghiệm cũ".
  - Plan note: "do not expand architecture complexity until simpler paths stall".

**Lý do hoãn:**

- Window=15 **quá ngắn** để attention thực sự khác biệt với last-step. Mean
  pool / attention pool trên 15 timestep gần như cho cùng kết quả.
- Lợi ích của attention chỉ rõ rệt khi window ≥ 30-60 (có thể distinguish
  giữa near-term momentum và mid-term context).
- Expected gain: `+0` đến `+0.0005` rel_score, không đáng để debug
  attention model.

**Điều kiện để revisit:**

- Khi mở rộng window lên 30+ ngày (sau khi rank/router edge stalls).
- Khi feature set chuyển sang multi-scale (vd: kết hợp window 5 + 15 + 30
  thông qua attention head).
- Khi muốn explain prediction (attention weights → "model đang nhìn ngày nào?").

**Đường đi đề xuất khi quay lại:**

1. Thay `return_sequences=True` ở **layer cuối** của `build_lstm_backbone` (
   khi `use_attention_pool=True`).
2. Thêm 1 layer attention pooling: `MultiHeadAttention(num_heads=2, key_dim=16)`
   self-attention trên 15 timestep, rồi `GlobalAveragePooling1D`.
3. A/B trên WF-CV: last-step (current) vs mean-pool vs attention-pool.
4. Drop nếu không cải thiện `≥ +0.001` val rel_score.

## Decision Log

- 2026-05-14: L2 và L3 hoãn lại sau khi đánh giá risk/reward với scope VN30
  hiện tại. Ưu tiên L4 (sample-weight ablation) vì target trực tiếp loss
  function quantile-based và effort thấp.
