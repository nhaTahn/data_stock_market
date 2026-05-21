from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FeatureScaler": ("src.models.training.scalers", "FeatureScaler"),
    "LocalTargetNormalizer": ("src.models.training.scalers", "LocalTargetNormalizer"),
    "TargetScaler": ("src.models.training.scalers", "TargetScaler"),
    "apply_feature_scaler": ("src.models.training.scalers", "apply_feature_scaler"),
    "apply_local_target_normalizer": ("src.models.training.scalers", "apply_local_target_normalizer"),
    "apply_target_scaler": ("src.models.training.scalers", "apply_target_scaler"),
    "balance_sample_weights_by_group": ("src.models.training.sample_weights", "balance_sample_weights_by_group"),
    "build_aux_plain_targets": ("src.models.training.targets", "build_aux_plain_targets"),
    "build_event_gated_sample_weights": ("src.models.training.sample_weights", "build_event_gated_sample_weights"),
    "build_event_gated_targets": ("src.models.training.targets", "build_event_gated_targets"),
    "build_inv_volatility_sample_weights": ("src.models.training.sample_weights", "build_inv_volatility_sample_weights"),
    "build_magnitude_sample_weights": ("src.models.training.sample_weights", "build_magnitude_sample_weights"),
    "build_sequence_dataset": ("src.models.training.datasets", "build_sequence_dataset"),
    "build_sign_magnitude_sample_weights": ("src.models.training.sample_weights", "build_sign_magnitude_sample_weights"),
    "build_sign_magnitude_targets": ("src.models.training.targets", "build_sign_magnitude_targets"),
    "extract_prediction_array": ("src.models.training.prediction", "extract_prediction_array"),
    "fit_attention_model": ("src.models.training.fitters", "fit_attention_model"),
    "fit_aux_plain_model": ("src.models.training.fitters", "fit_aux_plain_model"),
    "fit_deep_head_model": ("src.models.training.fitters", "fit_deep_head_model"),
    "fit_event_gated_model": ("src.models.training.fitters", "fit_event_gated_model"),
    "fit_hetero_model": ("src.models.training.fitters", "fit_hetero_model"),
    "fit_skip_model": ("src.models.training.fitters", "fit_skip_model"),
    "fit_feature_scaler": ("src.models.training.scalers", "fit_feature_scaler"),
    "fit_local_target_normalizer": ("src.models.training.scalers", "fit_local_target_normalizer"),
    "fit_model": ("src.models.training.fitters", "fit_model"),
    "fit_pcie_lite_model": ("src.models.training.fitters", "fit_pcie_lite_model"),
    "fit_quantile_model": ("src.models.training.fitters", "fit_quantile_model"),
    "fit_signal_attention_model": ("src.models.training.fitters", "fit_signal_attention_model"),
    "fit_sign_magnitude_model": ("src.models.training.fitters", "fit_sign_magnitude_model"),
    "fit_target_scaler": ("src.models.training.scalers", "fit_target_scaler"),
    "inverse_local_target_normalizer": ("src.models.training.scalers", "inverse_local_target_normalizer"),
    "inverse_target_scaler_values": ("src.models.training.scalers", "inverse_target_scaler_values"),
    "predict": ("src.models.training.prediction", "predict"),
    "primary_target_array": ("src.models.training.targets", "primary_target_array"),
    "set_global_seed": ("src.models.training.seeds", "set_global_seed"),
    "split_frame_by_date": ("src.models.training.datasets", "split_frame_by_date"),
    "split_sequence_dataset": ("src.models.training.datasets", "split_sequence_dataset"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
