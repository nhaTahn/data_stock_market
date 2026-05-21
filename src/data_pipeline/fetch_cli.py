from __future__ import annotations

import argparse


def add_fetch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--market",
        type=str,
        default="VN",
        choices=["VN", "US", "JP", "KR", "HK", "ALL"],
        help="Thị trường cần cào dữ liệu (VN, US, JP, KR, HK, ALL)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cào dữ liệu lịch sử cổ phiếu.")
    add_fetch_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def fetch_command(args: argparse.Namespace) -> None:
    from src.data_pipeline.fetch_data import fetch_all_market_data

    print("=================================================================")
    print("⬇️  BẮT ĐẦU CÀO DỮ LIỆU THỊ TRƯỜNG ⬇️")
    print(f"🌍 THỊ TRƯỜNG LỰA CHỌN: {args.market}")
    print("=================================================================\n")
    
    if args.market == 'ALL':
        markets = ['VN', 'US', 'JP', 'KR', 'HK']
    else:
        markets = [args.market]
        
    for m in markets:
        print(f"\n[GET] Đang lấy dữ liệu thị trường: {m}...")
        fetch_all_market_data(start_date='2010-01-01', output_dir='data/raw/', market=m)
        
    print("\n=================================================================")
    print("✅ HOÀN TẤT TẢI DỮ LIỆU TOÀN BỘ. Dữ liệu thô đặt tại thư mục `data/raw/`")
    print("=================================================================")


def main(argv: list[str] | None = None) -> None:
    fetch_command(parse_args(argv))


if __name__ == "__main__":
    main()
