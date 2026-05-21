from __future__ import annotations


BASELINE_MODEL_NAMES = ("linear_regression", "arima", "fischer_krauss")
REPORT_FAMILY_PREFIXES = (
    "lstm",
    "lstm_quantile",
    "lstm_signmag",
    "lstm_aux_plain",
    "lstm_hetero",
    "lstm_skip",
    "lstm_deep_head",
    "lstm_attention",
    "lstm_signal",
    "lstm_pcie_lite",
    "lstm_event",
)


def _strictly_longer_prefixes(prefix: str) -> tuple[str, ...]:
    """Return REPORT_FAMILY_PREFIXES entries that are strictly more specific than `prefix`."""
    return tuple(
        other for other in REPORT_FAMILY_PREFIXES
        if other != prefix and other.startswith(prefix) and len(other) > len(prefix)
    )


def select_report_model_names(model_names: list[str]) -> list[str]:
    available = set(model_names)
    selected: list[str] = []

    for baseline_name in BASELINE_MODEL_NAMES:
        if baseline_name in available:
            selected.append(baseline_name)

    for prefix in REPORT_FAMILY_PREFIXES:
        # Preference order:
        # 1. mean_ensemble  (mean of all seeds — reduces seed-luck variance, R6 default)
        # 2. best_by_val    (legacy default — kept for backward comparison)
        # 3. raw family     (single-seed runs only)
        preferred = [f"{prefix}_mean_ensemble", f"{prefix}_best_by_val", prefix]
        added = False
        for candidate in preferred:
            if candidate in available:
                selected.append(candidate)
                added = True
                break
        if not added:
            # Fallback: pick a name starting with `prefix` but NOT belonging to a
            # more specific family (e.g. when prefix="lstm" we must not pick
            # "lstm_signmag_*" — that is owned by the "lstm_signmag" iteration).
            deeper = _strictly_longer_prefixes(prefix)
            fallback = sorted(
                name
                for name in available
                if name.startswith(prefix)
                and "_seed_" not in name
                and not any(name.startswith(d) for d in deeper)
            )
            if fallback:
                selected.append(fallback[0])

    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name in available and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
