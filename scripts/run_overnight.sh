#!/bin/bash
set -e

# Khóa hiển thị cảnh báo từ TF tránh ngập màn hình
export TF_CPP_MIN_LOG_LEVEL=2

echo "==========================================="
echo "🌙 BẮT ĐẦU CHUỖI NHIỆM VỤ CHẠY QUA ĐÊM 🌙"
echo "==========================================="

echo ""
# echo "[1/4] Đang lấy dữ liệu thị trường mới nhất..."
# venv/bin/python scripts/run_fetch.py --market VN

# echo ""
# echo "[2/4] Đang biên dịch lại chất lượng Dataset..."
# venv/bin/python scripts/run_build_dataset.py --market VN

echo ""
echo "Đang rà soát Feature tốt nhất cho toàn bộ thị trường..."
echo "⚠️ Lưu ý: Quá trình này sẽ tốn vài giờ để quét tất cả các mã."
venv/bin/python src/models/run_all_vn_feature_searches.py


echo ""
echo "==========================================="
echo "✅ HOÀN TẤT TOÀN BỘ!"
echo "Báo cáo đã xuất tại: data/processed/assets/data_info_vn/history/training_runs/"
echo "==========================================="
