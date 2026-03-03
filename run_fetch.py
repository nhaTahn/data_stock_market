import argparse
from src.data_pipeline.fetch_data import fetch_all_market_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chỉ tải dữ liệu lịch sử cổ phiếu (Không chạy Machine Learning).')
    parser.add_argument('--market', type=str, default='VN', choices=['VN', 'US', 'JP', 'ALL'], help='Thị trường cần cào dữ liệu (VN, US, JP, ALL)')
    args = parser.parse_args()

    print("=================================================================")
    print("⬇️  BẮT ĐẦU CÀO DỮ LIỆU THỊ TRƯỜNG ⬇️")
    print(f"🌍 THỊ TRƯỜNG LỰA CHỌN: {args.market}")
    print("=================================================================\n")
    
    if args.market == 'ALL':
        markets = ['VN', 'US', 'JP']
    else:
        markets = [args.market]
        
    for m in markets:
        print(f"\n[GET] Đang lấy dữ liệu thị trường: {m}...")
        fetch_all_market_data(start_date='2010-01-01', output_dir='data/', market=m)
        
    print("\n=================================================================")
    print("✅ HOÀN TẤT TẢI DỮ LIỆU TOÀN BỘ. Dữ liệu thô đặt tại thư mục `data/`")
    print("=================================================================")
