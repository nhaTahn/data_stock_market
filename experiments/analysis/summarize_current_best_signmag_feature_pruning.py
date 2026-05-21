from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
DEFAULT_BATCH_ROOT = RUN_ROOT / "reports" / "feature_pruning"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the narrow broad-VN30 signmag feature-pruning experiment."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest JSON written by run_current_best_signmag_feature_pruning.py.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_metric(metrics: dict, model_name: str, split_name: str, key: str) -> float | None:
    split_metrics = metrics.get(model_name, {}).get(split_name, {})
    value = split_metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_evaluation_metric(evaluation_summary: dict, model_name: str, split_name: str, key: str) -> float | None:
    split_summary = evaluation_summary.get(model_name, {}).get(split_name, {})
    value = split_summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_quartile_equity(evaluation_summary: dict, model_name: str, split_name: str) -> float | None:
    quartile = (evaluation_summary.get(model_name, {}).get(split_name) or {}).get("quartile_long_short") or {}
    value = quartile.get("final_equity")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_summary_rows(manifest: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in manifest.get("cases", []):
        run_name = str(case["run_name"])
        core_dir = RUN_ROOT / run_name / "reports" / "core"
        if not core_dir.exists():
            rows.append(
                {
                    "case_name": case["case_name"],
                    "run_name": run_name,
                    "feature_count": int(case["feature_count"]),
                    "removed_groups": ",".join(case.get("removed_groups", [])),
                    "status": "missing_run",
                }
            )
            continue

        metrics = load_json(core_dir / "metrics.json")
        evaluation_summary = load_json(core_dir / "evaluation_summary.json")
        family_summary = load_json(core_dir / "family_selection_summary.json")

        plain_best = str((family_summary.get("lstm") or {}).get("best_by_val", ""))
        signmag_best = str((family_summary.get("lstm_signmag") or {}).get("best_by_val", ""))
        plain_report_model = "lstm_best_by_val" if "lstm_best_by_val" in metrics else plain_best
        signmag_report_model = "lstm_signmag_best_by_val" if "lstm_signmag_best_by_val" in metrics else signmag_best

        plain_val_rel = _safe_metric(metrics, plain_report_model, "val", "rel_score") if plain_report_model else None
        signmag_val_rel = _safe_metric(metrics, signmag_report_model, "val", "rel_score") if signmag_report_model else None

        best_family = "lstm"
        best_model = plain_report_model
        best_val_rel = plain_val_rel
        if signmag_val_rel is not None and (best_val_rel is None or signmag_val_rel > best_val_rel):
            best_family = "lstm_signmag"
            best_model = signmag_report_model
            best_val_rel = signmag_val_rel

        rows.append(
            {
                "case_name": case["case_name"],
                "run_name": run_name,
                "feature_count": int(case["feature_count"]),
                "removed_groups": ",".join(case.get("removed_groups", [])),
                "status": "ok",
                "plain_best_seed": plain_best,
                "plain_best_model": plain_report_model,
                "plain_val_rel_score": plain_val_rel,
                "plain_val_directional_accuracy": _safe_metric(metrics, plain_report_model, "val", "directional_accuracy") if plain_report_model else None,
                "plain_val_error_q2": _safe_evaluation_metric(evaluation_summary, plain_report_model, "val", "error_q2") if plain_report_model else None,
                "plain_val_error_q8": _safe_evaluation_metric(evaluation_summary, plain_report_model, "val", "error_q8") if plain_report_model else None,
                "plain_val_quartile_equity": _safe_quartile_equity(evaluation_summary, plain_report_model, "val") if plain_report_model else None,
                "signmag_best_seed": signmag_best,
                "signmag_best_model": signmag_report_model,
                "signmag_val_rel_score": signmag_val_rel,
                "signmag_val_directional_accuracy": _safe_metric(metrics, signmag_report_model, "val", "directional_accuracy") if signmag_report_model else None,
                "signmag_val_error_q2": _safe_evaluation_metric(evaluation_summary, signmag_report_model, "val", "error_q2") if signmag_report_model else None,
                "signmag_val_error_q8": _safe_evaluation_metric(evaluation_summary, signmag_report_model, "val", "error_q8") if signmag_report_model else None,
                "signmag_val_quartile_equity": _safe_quartile_equity(evaluation_summary, signmag_report_model, "val") if signmag_report_model else None,
                "winner_family": best_family,
                "winner_model": best_model,
                "winner_val_rel_score": best_val_rel,
            }
        )

    rows.sort(
        key=lambda item: (
            item.get("status") != "ok",
            -(item.get("winner_val_rel_score") if isinstance(item.get("winner_val_rel_score"), (int, float)) else float("-inf")),
            str(item["case_name"]),
        )
    )
    return rows


def write_summary_outputs(rows: list[dict[str, object]], batch_dir: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_columns = [
        "case_name",
        "run_name",
        "feature_count",
        "removed_groups",
        "status",
        "plain_best_seed",
        "plain_best_model",
        "plain_val_rel_score",
        "plain_val_directional_accuracy",
        "plain_val_error_q2",
        "plain_val_error_q8",
        "plain_val_quartile_equity",
        "signmag_best_seed",
        "signmag_best_model",
        "signmag_val_rel_score",
        "signmag_val_directional_accuracy",
        "signmag_val_error_q2",
        "signmag_val_error_q8",
        "signmag_val_quartile_equity",
        "winner_family",
        "winner_model",
        "winner_val_rel_score",
    ]

    summary_csv = batch_dir / "summary.csv"
    summary_json = batch_dir / "summary.json"
    summary_md = batch_dir / "summary.md"

    import csv

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Broad VN30 Signmag Feature-Pruning Summary",
        "",
        "| Case | Feature count | Removed groups | Plain val rel_score | Signmag val rel_score | Signmag q2/q8 | Signmag quartile equity | Winner |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        plain_rel = row.get("plain_val_rel_score")
        signmag_rel = row.get("signmag_val_rel_score")
        signmag_q2 = row.get("signmag_val_error_q2")
        signmag_q8 = row.get("signmag_val_error_q8")
        signmag_equity = row.get("signmag_val_quartile_equity")
        plain_text = f"{float(plain_rel):+.4f}" if isinstance(plain_rel, (int, float)) else "-"
        signmag_text = f"{float(signmag_rel):+.4f}" if isinstance(signmag_rel, (int, float)) else "-"
        q_text = (
            f"{float(signmag_q2):+.4f} / {float(signmag_q8):+.4f}"
            if isinstance(signmag_q2, (int, float)) and isinstance(signmag_q8, (int, float))
            else "-"
        )
        equity_text = f"{float(signmag_equity):.3f}" if isinstance(signmag_equity, (int, float)) else "-"
        winner_text = row.get("winner_model") or row.get("status", "-")
        lines.append(
            f"| `{row['case_name']}` | {row['feature_count']} | `{row.get('removed_groups', '') or '-'}` | {plain_text} | {signmag_text} | {q_text} | {equity_text} | `{winner_text}` |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_json(args.manifest)
    rows = build_summary_rows(manifest)
    write_summary_outputs(rows, args.manifest.parent)
    print(json.dumps({"manifest_path": str(args.manifest), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
