import os
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
import glob

def get_data_path(symbol):
    """
    Tự động tìm kiếm file CSV của một mã chứng khoán trong thư mục data/ và các thư mục con (VN, US, JP...).
    """
    matches = glob.glob(f"data/**/{symbol}.csv", recursive=True)
    if matches:
        return matches[0]
    return f"data/{symbol}.csv"



# Danh sách 30 mã chứng khoán rổ VN30
VN30_TICKERS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# Đọc list các mã chứng khoán (100 mã)
def load_market_list(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            raw_text = f.read().replace(',', ' ').replace('\n', ' ')
            tokens = [t.strip().upper() for t in raw_text.split() if t.strip()]
            return list(set(tokens))
    return []

def fetch_stock_data(symbol='ACB', start_date='2010-01-01', output_dir='data/'):
    """Tự động tải dữ liệu lịch sử của 1 mã bất kỳ và format theo chuẩn schema bằng vnstock3."""
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, f"{symbol}.csv")
    
    # Kiểm tra xem file đã có chưa, nếu có thì chỉ tải tiếp từ ngày cuối cùng (tiết kiệm thời gian)
    existing_df = None
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if not existing_df.empty and 'Date' in existing_df.columns:
                max_date = str(existing_df['Date'].max())
                if max_date >= end_date:
                    return existing_df
                # Chỉ lấy dữ liệu từ ngày mới nhất có trong file
                start_date = max_date
        except Exception as e:
            print(f"Lỗi đọc file cũ {output_path}: {e}")
    
    # 1. Tải dữ liệu từ vnstock3
    try:
        from vnstock import Vnstock
        
        # Khởi tạo kết nối tới TCBS để lấy dữ liệu chuẩn
        stock = Vnstock().stock(symbol=symbol, source='TCBS')
        df_raw = stock.quote.history(start=start_date, end=end_date)
        
        if df_raw is None or df_raw.empty:
            print(f"[!] Bỏ qua {symbol}: Không lấy được dữ liệu.")
            return existing_df
            
        # 2. Đổi tên cột chuẩn hóa theo yêu cầu
        rename_mapping = {
            'time': 'Date',           # Dùng cột này để làm ngày giao dịch
            'volume': 'volume_match'
        }
        df = df_raw.rename(columns=rename_mapping)
        
        # 3. Tính toán hoặc tạo các cột còn thiếu
        df['code'] = symbol
            
        # [QUAN TRỌNG]: Dữ liệu trả về từ TCBS mặc định là giá ĐÃ ĐIỀU CHỈNH
        # Nên ta gán nó vào cột adjust
        df['adjust'] = df['close']
            
        if 'value_match' not in df.columns:
            df['value_match'] = df['close'] * df['volume_match']
            
        # Đảm bảo cột Date luôn là string (YYYY-MM-DD)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
        # Đảm bảo có đủ cột
        final_columns = ['Date', 'code', 'high', 'low', 'open', 'close', 'adjust', 'volume_match', 'value_match']
        for col in final_columns:
            if col not in df.columns:
                df[col] = 0
                
        df = df[final_columns]
        
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Hợp nhất với dữ liệu cũ nếu có
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df])
            df.drop_duplicates(subset=['Date', 'code'], keep='last', inplace=True)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        # 4. Lưu ra file CSV
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return df

    except ImportError:
        print("[!] Không tìm thấy thư viện vnstock. Hãy chạy lệnh: !pip install vnstock")
        return existing_df
    except Exception as e:
        print(f"[!] Lỗi khi tải mã {symbol}: {e}")
        return None

def fetch_yfinance_data(symbol='AAPL', start_date='2010-01-01', output_dir='data/'):
    """Tải và chuẩn hóa dữ liệu quốc tế từ Yahoo Finance sang định dạng tương thích VNStock."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, f"{symbol}.csv")
    
    existing_df = None
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if not existing_df.empty and 'Date' in existing_df.columns:
                max_date = str(existing_df['Date'].max())
                if max_date >= end_date:
                    return existing_df
                # Fetch từ max_date
                start_date = max_date
        except Exception as e:
            print(f"Lỗi đọc file cũ {output_path}: {e}")
            
    try:
        # Tải Data từ Yahoo
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            print(f"[!] Bỏ qua {symbol} (YF): Không có dữ liệu.")
            return existing_df
            
        # Chuẩn hóa Dataset cho khớp 100% với form của VNStock
        df.reset_index(inplace=True)
        # Sàn Mỹ có timezone ở cột Date, cần convert sang YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['code'] = symbol
        
        # Đổi tên cột
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume_match'
        }, inplace=True)
        
        # Tạo thêm các cột mock giống với hệ VNStock để Pipeline trơn tru
        df['adjust'] = df['close']
        df['value_match'] = df['close'] * df['volume_match']
        
        final_columns = ['Date', 'code', 'high', 'low', 'open', 'close', 'adjust', 'volume_match', 'value_match']
        df = df[final_columns]
        
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Hợp nhất với dữ liệu cũ nếu có
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df])
            df.drop_duplicates(subset=['Date', 'code'], keep='last', inplace=True)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        # Lưu file
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
        
    except Exception as e:
        print(f"[!] Lỗi khi tải mã quốc tế {symbol}: {e}")
        return None

def fetch_all_market_data(start_date='2010-01-01', output_dir='data/', market='VN'):
    """Vòng lặp tải dữ liệu dựa trên Market được chọn (VN, US, JP, ALL)"""
    
    # 1. Định vị danh sách cổ phiếu
    target_tickers = []
    is_foreign = market.upper() in ['US', 'JP', 'KR', 'HK']
    
    if market.upper() == 'VN':
        target_tickers = load_market_list('market_lists/vn100.txt') or VN30_TICKERS.copy()
        output_dir = 'data/VN/'
    elif market.upper() == 'US':
        target_tickers = load_market_list('market_lists/us100.txt')
        output_dir = 'data/US/'
    elif market.upper() == 'JP':
        target_tickers = load_market_list('market_lists/jp50.txt')
        output_dir = 'data/JP/'
    elif market.upper() == 'KR':
        target_tickers = load_market_list('market_lists/kr50.txt')
        output_dir = 'data/KR/'
    elif market.upper() == 'HK':
        target_tickers = load_market_list('market_lists/hk50.txt')
        output_dir = 'data/HK/'
            

    print(f"Bắt đầu tải dữ liệu lịch sử nhóm {market.upper()} (Tổng: {len(target_tickers)} mã) từ {start_date}...")
    success_count = 0
    failure_list = []
    
    for i, symbol in enumerate(target_tickers, 1):
        print(f"[{i}/{len(target_tickers)}] Đang tải {symbol}...", end=" ")
        
        # 2. Định tuyến Fetcher (VNStock vs Yahoo Finance)
        # Tạm định nghĩa: US, JP dùng YF. Nếu là custom Watchlist ngoại lệ mà mã có dấu '.T' hoặc chữ số -> chuyển sang YF.
        if is_foreign or ('.' in symbol) or (symbol.isalpha() and len(symbol) > 3) or symbol.startswith('^'):
            df = fetch_yfinance_data(symbol, start_date, output_dir)
        else:
            df = fetch_stock_data(symbol, start_date, output_dir)
            
        if df is not None:
            print(f"OK ({len(df)} dòng).")
            success_count += 1
        else:
            failure_list.append(symbol)
            
        time.sleep(4)
        
    print(f"\n[HOÀN TẤT TẢI DATA] Thành công {success_count}/{len(target_tickers)} mã.")
    if failure_list:
        print(f"Các mã gặp lỗi: {', '.join(failure_list)}")
        
    return target_tickers

if __name__ == "__main__":
    fetch_all_market_data()
