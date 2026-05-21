# Gold Index

- `current_best_vn30`: cụm VIN (`VIC,VHM,VRE`), rel_score test `0.043404`.
- `current_best_vn30_broad`: broad VN30 29 mã, rel_score test `0.001336`.
- `current_best_vn100`: F&B committee (`KDC,SAB,SBT,VNM`), committee test rel_score `0.049321`.
- `feature_glossary_current_models.md`: chú thích feature đang dùng, ý nghĩa và công thức tính.
- `reports/current_best_time_windows_20260424/`: histogram theo 3 giai đoạn thời gian cho các package `current_best_*`.

Ghi chú:

- `current_best_vn30` là edge tốt nhất hiện còn giữ trong universe VN30, nhưng là cluster-level.
- `current_best_vn30_broad` là gói broad-panel VN30 để báo cáo trung thực về khả năng dự báo toàn rổ.
- `current_best_vn100` hiện là package VN100-related tốt nhất còn giữ sau cleanup; broad VN100 run không còn được retain.
