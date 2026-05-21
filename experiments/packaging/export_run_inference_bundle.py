from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from src.models.training.datasets import build_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame  # noqa: E402
from src.models.training.scalers import fit_local_target_normalizer  # noqa: E402
from src.utils.features import ensure_paper_features  # noqa: E402


@dataclass(frozen=True)
class ExportModelSpec:
    source_alias: str | None
    resolved_model_name: str
    model_family: str
    checkpoint_name: str
    prediction_key: str | int | tuple[int, ...] | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a saved training run into a Colab-friendly inference bundle.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="best_overall_by_val",
        help=(
            "Model selector. Supported values: best_overall_by_val, lstm_best_by_val, "
            "lstm_signmag_best_by_val, or a concrete seed model like lstm_seed_52."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_alias_to_model_name(selector: str, summary: dict[str, object], metrics: dict[str, object]) -> tuple[str | None, str]:
    if selector == "best_overall_by_val":
        candidates: list[tuple[str, float]] = []
        alias_map = {
            "lstm_best_by_val": (((summary.get("lstm") or {}).get("best_by_val"))),
            "lstm_signmag_best_by_val": (((summary.get("lstm_signmag") or {}).get("best_by_val"))),
        }
        for alias, seed_name in alias_map.items():
            if not seed_name:
                continue
            metric = metrics.get(alias)
            if not isinstance(metric, dict):
                continue
            val = metric.get("val")
            if not isinstance(val, dict) or "rel_score" not in val:
                continue
            candidates.append((alias, float(val["rel_score"])))
        if not candidates:
            raise ValueError("Could not resolve best_overall_by_val from family_selection_summary.json and metrics.json.")
        best_alias = max(candidates, key=lambda item: item[1])[0]
        return resolve_alias_to_model_name(best_alias, summary, metrics)

    alias_lookup = {
        "lstm_best_by_val": ("lstm", "best_by_val"),
        "lstm_signmag_best_by_val": ("lstm_signmag", "best_by_val"),
    }
    if selector in alias_lookup:
        family_key, slot_key = alias_lookup[selector]
        resolved = (((summary.get(family_key) or {}).get(slot_key)))
        if not resolved:
            raise ValueError(f"Could not resolve alias {selector} from family_selection_summary.json.")
        return selector, str(resolved)
    return None, selector


def resolve_export_spec(selector: str, summary: dict[str, object], metrics: dict[str, object]) -> ExportModelSpec:
    source_alias, resolved_model_name = resolve_alias_to_model_name(selector, summary, metrics)
    if resolved_model_name.startswith("lstm_signmag_seed_"):
        seed = resolved_model_name.removeprefix("lstm_signmag_seed_")
        return ExportModelSpec(
            source_alias=source_alias,
            resolved_model_name=resolved_model_name,
            model_family="signmag",
            checkpoint_name=f"model_signmag_seed_{seed}.keras",
            prediction_key="signed_prediction",
        )
    if resolved_model_name.startswith("lstm_seed_"):
        seed = resolved_model_name.removeprefix("lstm_seed_")
        return ExportModelSpec(
            source_alias=source_alias,
            resolved_model_name=resolved_model_name,
            model_family="plain",
            checkpoint_name=f"model_seed_{seed}.keras",
            prediction_key=None,
        )
    raise ValueError(
        f"Unsupported model selector '{selector}' resolved to '{resolved_model_name}'. "
        "Only single-seed plain/signmag models are exportable."
    )


def build_local_target_normalizer_payload(config: dict[str, object]) -> dict[str, object] | None:
    target_normalizer = config.get("target_normalizer")
    if not target_normalizer:
        return None
    data_path = Path(str(config["data_path"]))
    stocks = str(config.get("stocks") or "")
    train_df = load_frame(data_path, stocks)
    if str(config.get("feature_phase", "none")) in {"paper_v1", "paper_denoise_v1"}:
        train_df = ensure_paper_features(train_df)

    alias = f"__target_normalizer__{target_normalizer}"
    if target_normalizer not in train_df.columns:
        raise ValueError(f"Target normalizer column missing from source dataset: {target_normalizer}")
    train_df[alias] = train_df[str(target_normalizer)].astype(float)
    _, _, meta_all = build_sequence_dataset(
        train_df,
        tuple(str(item) for item in config["feature_columns"]),
        str(config["target_column"]),
        int(config["window_size"]),
        extra_meta_columns=(alias,),
        sequence_normalization=str(config.get("sequence_normalization", "none")),
    )
    if meta_all.empty:
        return None
    train_mask = meta_all["Date"] <= config["train_end_date"]
    scale_values = meta_all.loc[train_mask, alias].to_numpy(dtype="float32")
    normalizer = fit_local_target_normalizer(scale_values, str(target_normalizer))
    return {
        "column": normalizer.column,
        "floor": float(normalizer.floor),
    }


def write_readme(output_dir: Path, run_dir: Path, spec: ExportModelSpec, metrics: dict[str, object]) -> None:
    val_rel_score = None
    if spec.source_alias and spec.source_alias in metrics:
        alias_metrics = metrics[spec.source_alias]
        if isinstance(alias_metrics, dict):
            val_metrics = alias_metrics.get("val")
            if isinstance(val_metrics, dict):
                val_rel_score = val_metrics.get("rel_score")
    lines = [
        "# Inference Bundle",
        "",
        f"- source_run: `{run_dir}`",
        f"- source_alias: `{spec.source_alias}`",
        f"- resolved_model_name: `{spec.resolved_model_name}`",
        f"- model_family: `{spec.model_family}`",
        f"- checkpoint_name: `{spec.checkpoint_name}`",
    ]
    if val_rel_score is not None:
        lines.append(f"- validation_rel_score: `{float(val_rel_score):+.6f}`")
    lines.extend(
        [
            "",
            "Bundle này dành cho inference trên Colab hoặc notebook, không phải bundle train lại.",
            "Để suy luận đúng, cần dùng cùng `source_config.json`, `feature_scaler.npz`, `target_scaler.npz` và checkpoint.",
        ]
    )
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "core").mkdir(parents=True, exist_ok=True)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    config = load_json(run_dir / "reports" / "core" / "config.json")
    metrics = load_json(run_dir / "reports" / "core" / "metrics.json")
    family_summary = load_json(run_dir / "reports" / "core" / "family_selection_summary.json")
    spec = resolve_export_spec(str(args.model), family_summary, metrics)
    local_target_normalizer = build_local_target_normalizer_payload(config)

    checkpoint_path = run_dir / spec.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    shutil.copy2(run_dir / "feature_scaler.npz", output_dir / "model" / "feature_scaler.npz")
    if (run_dir / "target_scaler.npz").exists():
        shutil.copy2(run_dir / "target_scaler.npz", output_dir / "model" / "target_scaler.npz")
    shutil.copy2(checkpoint_path, output_dir / "model" / spec.checkpoint_name)
    shutil.copy2(run_dir / "reports" / "core" / "config.json", output_dir / "core" / "source_config.json")
    shutil.copy2(run_dir / "reports" / "core" / "metrics.json", output_dir / "core" / "source_metrics.json")
    shutil.copy2(
        run_dir / "reports" / "core" / "family_selection_summary.json",
        output_dir / "core" / "family_selection_summary.json",
    )

    manifest = {
        "source_run_dir": str(run_dir),
        "source_alias": spec.source_alias,
        "resolved_model_name": spec.resolved_model_name,
        "model_family": spec.model_family,
        "checkpoint_name": spec.checkpoint_name,
        "prediction_key": spec.prediction_key,
        "local_target_normalizer": local_target_normalizer,
    }
    save_json(output_dir / "core" / "bundle_manifest.json", manifest)
    write_readme(output_dir, run_dir, spec, metrics)
    print(json.dumps({"output_dir": str(output_dir), "bundle_manifest": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
