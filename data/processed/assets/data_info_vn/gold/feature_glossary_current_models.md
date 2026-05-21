# Feature Glossary For Current Gold Models

File này tóm tắt các feature đang dùng trong các bundle giữ lại ở `gold`, kèm ý nghĩa và công thức tính gần đúng theo code thật trong [features.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/utils/features.py).

## Ký hiệu chung

- `O_t, H_t, L_t, C_t`: open, high, low, close ngày `t`
- `A_t`: adjusted close ngày `t`
- `V_t`: `volume_match` ngày `t`
- `Val_t`: `value_match` ngày `t`
- `ret_t`: `pct_change(A_t)` theo từng mã
- `MA_n(x)_t`: rolling mean `n` phiên của biến `x`
- `STD_n(x)_t`: rolling std `n` phiên của biến `x`
- `MAX_n(x)_t`, `MIN_n(x)_t`: rolling max/min `n` phiên

## Bundle Mapping

- `current_best_vn30`
  Dùng bộ feature của `best_vin_cluster_lstm_attention_20260417`
- `current_best_vn30_broad`
  Dùng bộ feature `paper_v1` của `best_vn30_broad_lstm_signmag_seed_52_20260412`
- `current_best_vn100`
  Là committee, không ăn raw feature trực tiếp
  `market sidecar` dùng feature set rộng của `source_run_config.json`
  `expert standalone` dùng feature set hẹp của `standalone_config.json`

## Current Best VN30 Cluster

- `vwap_gap`
  Chênh lệch giữa close và VWAP proxy trong ngày
  `vwap_proxy_t = Val_t / V_t`
  `vwap_gap_t = C_t / vwap_proxy_t - 1`

- `vwap_gap_20`
  Chênh lệch giữa close và rolling VWAP 20 phiên
  `vwap_20_t = SUM_20(Val) / SUM_20(V)`
  `vwap_gap_20_t = C_t / vwap_20_t - 1`

- `bb_width`
  Độ rộng Bollinger Band 20 phiên, đo mức nén/mở rộng biến động
  `bb_mid_t = MA_20(A)_t`
  `bb_std_t = STD_20(A)_t`
  `bb_upper_t = bb_mid_t + 2 * bb_std_t`
  `bb_lower_t = bb_mid_t - 2 * bb_std_t`
  `bb_width_t = (bb_upper_t - bb_lower_t) / bb_mid_t`

- `volatility_20`
  Độ biến động rolling 20 phiên của adjusted return
  `volatility_20_t = STD_20(ret)_t`

- `gap_open`
  Gap giữa open hôm nay và close hôm trước
  `gap_open_t = O_t / C_{t-1} - 1`

- `intraday_return`
  Return trong ngày
  `intraday_return_t = C_t / O_t - 1`

- `close_position`
  Vị trí đóng cửa trong biên độ nến
  `close_position_t = (C_t - L_t) / (H_t - L_t)`

- `obv_change`
  Tốc độ thay đổi của OBV
  `OBV_t = SUM(sign(close_return_tau) * V_tau)`
  `obv_change_t = OBV_t / OBV_{t-1} - 1`

- `momentum_5`
  Xung lượng 5 phiên
  `momentum_5_t = A_t / A_{t-5} - 1`

- `momentum_20`
  Xung lượng 20 phiên
  `momentum_20_t = A_t / A_{t-20} - 1`

- `above_ma_200`
  Cờ xu hướng dài hạn
  `above_ma_200_t = 1 nếu A_t > MA_200(A)_t, ngược lại 0`

- `lower_shadow`
  Bóng nến dưới, chuẩn hóa theo close
  `lower_shadow_t = (min(O_t, C_t) - L_t) / C_t`

- `upper_shadow`
  Bóng nến trên, chuẩn hóa theo close
  `upper_shadow_t = (H_t - max(O_t, C_t)) / C_t`

- `ma_200_gap`
  Khoảng cách tới MA200
  `ma_200_gap_t = A_t / MA_200(A)_t - 1`

- `volume_ratio_20`
  Tỷ lệ volume hiện tại so với trung bình 20 phiên
  `volume_ratio_20_t = V_t / MA_20(V)_t`

- `rsi_14`
  RSI 14 phiên
  `gain_t = max(Delta A_t, 0)`
  `loss_t = max(-Delta A_t, 0)`
  `RS_t = MA_14(gain)_t / MA_14(loss)_t`
  `RSI_t = 100 - 100 / (1 + RS_t)`

- `macd_hist`
  Histogram MACD
  `MACD_t = EMA_12(A)_t - EMA_26(A)_t`
  `Signal_t = EMA_9(MACD)_t`
  `macd_hist_t = MACD_t - Signal_t`

- `rolling_max_20_gap`
  Mức cách xa đỉnh 20 phiên
  `rolling_max_20_gap_t = A_t / MAX_20(A)_t - 1`

- `wyckoff_phase_60d`
  Vị trí giá hiện tại trong range 60 phiên
  `wyckoff_phase_60d_t = (C_t - MIN_60(L)_t) / (MAX_60(H)_t - MIN_60(L)_t)`

- `effort_result_ratio`
  Tỷ lệ effort/result theo Wyckoff-VSA
  `spread_t = H_t - L_t`
  `vol_norm_t = V_t / MAX_20(V)_t`
  `effort_result_ratio_t = vol_norm_t / spread_t`

- `buying_pressure`
  Áp lực mua trong nến
  `buying_pressure_t = ((C_t - L_t) / (H_t - L_t)) * vol_norm_t`

- `selling_pressure`
  Áp lực bán trong nến
  `selling_pressure_t = ((H_t - C_t) / (H_t - L_t)) * vol_norm_t`

- `alpha_sector`
  Return vượt/trượt so với sector mean leave-one-out
  `sector_return_ex_self_t = (SUM_sector(ret)_t - ret_t) / (N_sector_t - 1)`
  `alpha_sector_t = ret_t - sector_return_ex_self_t`

- `day_of_week`
  Thứ trong tuần dạng số
  `0 = Monday, ..., 4 = Friday`

- `vingroup_momentum`
  Return trung bình theo ngày của nhóm `VIC,VHM,VRE`
  `vingroup_momentum_t = mean(ret_t của VIC,VHM,VRE)`

- `vnindex_return`
  Proxy breadth/market return của toàn cross-section trong dataset
  `vnindex_return_t = mean(ret_t của các mã trong dataset)`

- `a_d_ratio`
  Advance/Decline ratio theo ngày
  `a_d_ratio_t = advancing_t / (declining_t + 1)`

## Current Best VN30 Broad

`paper_v1` giữ lại các feature dưới đây.

- `open_level_20`
  Mức lệch của open so với MA20(open)
  `open_level_20_t = O_t / MA_20(O)_t - 1`

- `high_level_20`
  `high_level_20_t = H_t / MA_20(H)_t - 1`

- `low_level_20`
  `low_level_20_t = L_t / MA_20(L)_t - 1`

- `close_level_20`
  `close_level_20_t = C_t / MA_20(C)_t - 1`

- `volume_level_20`
  `volume_level_20_t = V_t / MA_20(V)_t - 1`

- `open_delta_1`
  Biến động 1 phiên của open
  `open_delta_1_t = O_t / O_{t-1} - 1`

- `high_delta_1`
  `high_delta_1_t = H_t / H_{t-1} - 1`

- `low_delta_1`
  `low_delta_1_t = L_t / L_{t-1} - 1`

- `close_delta_1`
  `close_delta_1_t = C_t / C_{t-1} - 1`

- `volume_delta_1`
  `volume_delta_1_t = V_t / V_{t-1} - 1`

- `intraday_return`
  `C_t / O_t - 1`

- `gap_open`
  `O_t / C_{t-1} - 1`

- `close_position`
  `(C_t - L_t) / (H_t - L_t)`

- `bb_width`
  `(bb_upper_t - bb_lower_t) / bb_mid_t`

- `volume_ratio_20`
  `V_t / MA_20(V)_t`

- `volatility_20`
  `STD_20(ret)_t`

- `momentum_5`
  `A_t / A_{t-5} - 1`

- `momentum_20`
  `A_t / A_{t-20} - 1`

- `macd_hist`
  `EMA_12(A)_t - EMA_26(A)_t - EMA_9(MACD)_t`

- `rsi_14`
  RSI 14 phiên

- `vnindex_return`
  Mean daily return của cross-section

- `vingroup_momentum`
  Mean daily return của `VIC,VHM,VRE`

- `a_d_ratio`
  `advancing / (declining + 1)`

- `day_of_week`
  Số thứ trong tuần

## Current Best VN100 Committee

Committee không trực tiếp học trên raw feature. Nó trộn dự báo từ:

- `market sidecar`
  feature set rộng:
  `volume_ratio_20, intraday_return, gap_open, close_position, upper_shadow, lower_shadow, momentum_5, momentum_20, volatility_20, ma_200_gap, rolling_max_20_gap, bb_width, vwap_gap, obv_change, macd_hist, effort_result_ratio, buying_pressure, selling_pressure, wyckoff_phase_60d, a_d_ratio, vingroup_momentum, vnindex_return, rsi_14, day_of_week`

- `expert standalone`
  feature set hẹp:
  `volume_ratio_20, close_position, lower_shadow, alpha_sector, vingroup_momentum, vnindex_return, a_d_ratio, day_of_week, rsi_14`

Các feature này dùng cùng định nghĩa/công thức ở hai section phía trên.

## Ghi chú dùng khi báo cáo

- Các feature có rolling window như `momentum_20`, `volatility_20`, `bb_width`, `volume_ratio_20` cần đủ lịch sử để ổn định.
- `alpha_sector` trong code hiện là leave-one-out mean, đã sửa để tránh leakage vòng lặp.
- `wyckoff_phase_60d`, `effort_result_ratio`, `buying_pressure`, `selling_pressure` là các feature nghiêng về cấu trúc dòng tiền và vị trí giá trong range.
- `paper_v1` thêm dual-view giữa `level_20` và `delta_1`, tức là vừa nhìn mức tuyệt đối so với rolling mean, vừa nhìn biến động phần trăm ngắn hạn.
