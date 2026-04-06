#!/usr/bin/env python3
import itertools
import subprocess
import json
import csv
from pathlib import Path
import sys

def main():
    lstm_units = [32, 64, 128]
    dropouts = [0.2, 0.3]
    lrs = [0.001, 0.0005]

    extra_args = sys.argv[1:]
    ROOT = Path(__file__).resolve().parents[1]

    run_names = []

    # Lặp qua tất cả 12 tổ hợp (3 * 2 * 2)
    for u, d, l in itertools.product(lstm_units, dropouts, lrs):
        run_name = f"lstm_units_{u}_dropout_{d}_lr_{l}"
        print("=" * 60)
        print(f"🚀 Bắt đầu chạy thử nghiệm: {run_name}")
        
        cmd = [
            "venv/bin/python", "scripts/run_train.py",
            "--lstm-units", str(u),
            "--dropout", str(d),
            "--lr", str(l),
            "--run-name", run_name
        ]
        
        # Nếu người dùng không chỉ định list cổ phiếu, mình sẽ truyền mặc định 18 mã BĐS
        # (Để có đủ data tránh việc fail như chạy 7 mã). 
        # Tuy nhiên do yêu cầu không fix cứng nên sẽ dùng params của user. Nếu không có thì fallback:
        if "--stocks" not in extra_args and "-s" not in extra_args:
            # Bạn có thể tuỳ chỉnh lại mã. Mình đang đặt sẵn list 18 mã tốt nhất hôm trước
            cmd += ["--stocks", "VHM,VIC,VRE,KBC,NVL,DIG,DXG,PDR,NLG,KDH,CEO,HDG,TCH,HDC,CRE,SJS,IJC,NBB", "--epochs", "20"]

        cmd += extra_args
        
        try:
            subprocess.run(cmd, check=True)
            run_names.append(run_name)
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy {run_name}: {e}")

    print("=" * 60)
    print("🎯 Hoàn thành tất cả các thử nghiệm! Đang tổng hợp kết quả...")
    aggregate_results(ROOT, run_names)

def aggregate_results(root: Path, run_names: list[str]):
    base_dir = root / "data/processed/assets/data_info_vn/history/training_runs"
    
    results = []
    
    for run_name in run_names:
        run_dir = base_dir / run_name
        metrics_file = run_dir / "metrics.json"
        
        if not metrics_file.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy {metrics_file}.")
            continue
            
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            
        lstm_metrics = metrics.get("lstm", {})
        lstm_test = lstm_metrics.get("test", {})
        lstm_val = lstm_metrics.get("val", {})
        
        parts = run_name.split("_")
        u = parts[2]
        d = parts[4]
        l = parts[6]
        
        row = {
            "run_name": run_name,
            "lstm_units": u,
            "dropout": d,
            "lr": l,
            "val_rel_score": lstm_val.get("rel_score", ""),
            "test_directional_accuracy": lstm_test.get("directional_accuracy", ""),
            "test_rel_score": lstm_test.get("rel_score", ""),
            "test_mse": lstm_test.get("mse", ""),
            "test_mae": lstm_test.get("mae", "")
        }
        results.append(row)
        
    out_csv = root / "experiment_results.csv"
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Đã ghi báo cáo tổng hợp siêu tham số tại: {out_csv}")
    else:
        print("Không có kết quả nào để tổng hợp.")

if __name__ == "__main__":
    main()
