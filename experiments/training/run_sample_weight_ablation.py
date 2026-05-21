"""Sample-weight ablation runner (L4 of plan).

Mục đích: so sánh 4 chiến lược sample_weight trên cùng config base, dùng
walk-forward CV để tránh chạm validation Apr-2020 → Nov-2022.

Các variant test:

| Variant | sample_weight_mode | sample_weight_balance_mode | Ghi chú |
| --- | --- | --- | --- |
| `none` | `none` | `none` | Baseline — uniform weights |
| `magnitude` | `magnitude` | `none` | Current default (upweight |y| lớn) |
| `magnitude_balance` | `magnitude` | `market` | Magnitude + balance across market group |
| `inv_volatility` | `inv_volatility` | `none` | L4 — downweight high-vol days |

Output: thư mục `reports/sample_weight_ablation/<output_name>/` với:
- `summary.md` — bảng comparison cuối
- per-variant subdirs (mỗi variant = một WF-CV run)

Cách chạy:

```
python experiments/training/run_sample_weight_ablation.py \\
    --output-name sweight_ablation_20260514 \\
    --wf-train-end 2020-03-31 \\
    --min-train-days 1500 \\
    --val-days 126 \\
    --step-days 126 \\
    --embargo-days 5 \\
    --max-folds 4 \\
    --lstm-seeds 42 \\
    --epochs 12 \\
    --window-size 15 \\
    --lstm-units 64,32 \\
    --dropout 0.05 \\
    --loss rel_score
```

(Có thể pass-through bất kỳ args nào của pipeline; ablation runner chỉ
override `--sample-weight-mode` và `--sample-weight-balance-mode` cho từng
variant.)

Smoke mode (kiểm tra plumbing, không train):

```
python experiments/training/run_sample_weight_ablation.py \\
    --output-name plumbing --smoke ...
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VARIANTS: list[dict[str, str]] = [
    {"name": "none", "sample_weight_mode": "none", "sample_weight_balance_mode": "none"},
    {"name": "magnitude", "sample_weight_mode": "magnitude", "sample_weight_balance_mode": "none"},
    {"name": "magnitude_balance", "sample_weight_mode": "magnitude", "sample_weight_balance_mode": "market"},
    {"name": "inv_volatility", "sample_weight_mode": "inv_volatility", "sample_weight_balance_mode": "none"},
]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="L4 sample-weight ablation via walk-forward CV.",
        allow_abbrev=False,
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "assets"
        / "data_info_vn"
        / "history"
        / "training_runs"
        / "reports"
        / "sample_weight_ablation",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
    )
    parser.add_argument(
        "--only-variants",
        type=str,
        default=None,
        help="Comma-separated subset of variant names (e.g. 'none,inv_volatility').",
    )
    return parser.parse_known_args(argv)


def run_variant(
    variant: dict[str, str],
    *,
    output_dir: Path,
    passthrough: list[str],
    python_bin: str,
    smoke: bool,
    dry_run: bool,
) -> Path:
    variant_name = variant["name"]
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        python_bin,
        str(ROOT / "experiments" / "training" / "run_walk_forward_lstm_search.py"),
        "--output-name",
        variant_name,
        "--output-root",
        str(output_dir),
        "--sample-weight-mode",
        variant["sample_weight_mode"],
        "--sample-weight-balance-mode",
        variant["sample_weight_balance_mode"],
    ]
    if smoke:
        cmd.append("--smoke")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(passthrough)

    print(f"\n=== Variant: {variant_name} ===")
    print(" ".join(cmd))
    if dry_run:
        return variant_dir / "summary.md"
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"  WARN: variant {variant_name} exited with code {result.returncode}")
    return variant_dir / "summary.md"


def read_variant_metrics(summary_md: Path) -> dict[str, object]:
    """Pull rel_score/IC aggregates from a WF runner's summary.md.

    Falls back to reading fold_metrics.csv directly if available.
    """
    record: dict[str, object] = {"summary_path": str(summary_md)}
    fold_metrics_csv = summary_md.parent / "fold_metrics.csv"
    if fold_metrics_csv.exists():
        try:
            import pandas as pd  # local import to keep CLI light

            df = pd.read_csv(fold_metrics_csv)
            if "val_rel_score" in df.columns:
                series = df["val_rel_score"].dropna()
                if not series.empty:
                    record["val_rel_score_mean"] = float(series.mean())
                    record["val_rel_score_median"] = float(series.median())
                    record["val_rel_score_std"] = float(series.std(ddof=1)) if len(series) >= 2 else 0.0
                    record["val_rel_score_min"] = float(series.min())
                    record["val_rel_score_max"] = float(series.max())
                    record["n_folds"] = int(len(series))
            if "val_mean_daily_ic" in df.columns:
                series = df["val_mean_daily_ic"].dropna()
                if not series.empty:
                    record["val_ic_mean"] = float(series.mean())
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"failed to parse fold_metrics.csv: {exc}"
    elif not summary_md.exists():
        record["error"] = "no summary.md or fold_metrics.csv found"
    return record


def write_ablation_summary(
    output_dir: Path,
    variant_metrics: list[dict[str, object]],
    args: argparse.Namespace,
    passthrough: list[str],
) -> Path:
    lines = [
        f"# Sample-Weight Ablation — {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        f"- output_name: `{args.output_name}`",
        f"- variants: `{len(variant_metrics)}`",
        f"- passthrough: `{' '.join(passthrough)}`",
        "",
        "## Comparison",
        "",
        "| variant | n_folds | val_rel_score mean | median | std | min | max | val_ic_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in variant_metrics:
        name = record.get("variant", "?")
        n = record.get("n_folds", "—")
        rel_mean = record.get("val_rel_score_mean")
        rel_med = record.get("val_rel_score_median")
        rel_std = record.get("val_rel_score_std")
        rel_min = record.get("val_rel_score_min")
        rel_max = record.get("val_rel_score_max")
        ic_mean = record.get("val_ic_mean")

        def _fmt(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.5f}"
            return str(v) if v is not None else "—"

        lines.append(
            f"| {name} | {n} | {_fmt(rel_mean)} | {_fmt(rel_med)} | {_fmt(rel_std)} | "
            f"{_fmt(rel_min)} | {_fmt(rel_max)} | {_fmt(ic_mean)} |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
        "- Chọn variant có `val_rel_score mean` cao nhất với `std` không vượt quá",
        "  baseline đáng kể (`std` cao = phụ thuộc fold, kém ổn định qua thời gian).",
        "- Nếu hai variant gần ngang nhau về mean, ưu tiên variant có `min` (worst-fold)",
        "  cao hơn — đây là proxy cho robustness qua regime.",
        "- Nếu `inv_volatility` cải thiện `val_rel_score` so với `magnitude`: "
        "khẳng định giả thuyết rằng current weighting overweight outlier ngày bão.",
        "",
        "## Next Step",
        "",
        f"Pick the winning variant, set it as `sample_weight_mode` (and `*_balance_mode`)",
        f"in `configs/lstm_config.json`, then proceed to L5 (final fit on full train +",
        f"hold-out evaluation).",
    ]

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    known, passthrough = parse_args(argv)
    output_dir = known.output_root / known.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    only = (
        {v.strip() for v in known.only_variants.split(",") if v.strip()}
        if known.only_variants
        else None
    )
    chosen = [v for v in VARIANTS if only is None or v["name"] in only]
    if not chosen:
        raise SystemExit(f"No variants selected. Available: {[v['name'] for v in VARIANTS]}")

    print(f"Ablation output dir: {output_dir}")
    print(f"Variants to run: {[v['name'] for v in chosen]}")

    variant_metrics: list[dict[str, object]] = []
    for variant in chosen:
        summary_md = run_variant(
            variant,
            output_dir=output_dir,
            passthrough=passthrough,
            python_bin=known.python_bin,
            smoke=known.smoke,
            dry_run=known.dry_run,
        )
        record = {"variant": variant["name"]}
        if not known.dry_run and not known.smoke:
            record.update(read_variant_metrics(summary_md))
        variant_metrics.append(record)

    (output_dir / "variant_metrics.json").write_text(
        json.dumps(variant_metrics, indent=2, default=str),
        encoding="utf-8",
    )
    write_ablation_summary(output_dir, variant_metrics, known, passthrough)
    print(f"\nDone. Comparison at {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
