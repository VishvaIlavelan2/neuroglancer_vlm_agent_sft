from __future__ import annotations

from typing import Any

from .reward_utils import compute_visibility_based_reward


ACTION_QUANTIZATION = {
    "delta_x": 50.0,
    "delta_y": 50.0,
    "delta_z": 10.0,
}


def _round_to_quantum(value: float, quantum: float) -> float:
    if quantum <= 0:
        raise ValueError(f"quantum must be positive, got {quantum}")
    return round(float(value) / quantum) * quantum


def quantize_action(action: dict[str, Any]) -> dict[str, Any]:
    quantized = dict(action)
    quantized["delta_x"] = _round_to_quantum(action.get("delta_x", 0.0), ACTION_QUANTIZATION["delta_x"])
    quantized["delta_y"] = _round_to_quantum(action.get("delta_y", 0.0), ACTION_QUANTIZATION["delta_y"])
    quantized["delta_z"] = _round_to_quantum(action.get("delta_z", 0.0), ACTION_QUANTIZATION["delta_z"])
    return quantized


def quantize_previous_outcome(previous_outcome: dict[str, Any]) -> dict[str, Any]:
    visibility_label = str(
        previous_outcome.get("visibility_label", previous_outcome.get("label", "unknown"))
    )
    z_delta = _round_to_quantum(
        previous_outcome.get("z_delta", 0.0),
        ACTION_QUANTIZATION["delta_z"],
    )
    if visibility_label in {"visible", "uncertain", "not_visible"}:
        reward = compute_visibility_based_reward(visibility_label, z_delta)
    else:
        reward = float(previous_outcome.get("reward", 0.0))
    return {
        **previous_outcome,
        "visibility_label": visibility_label,
        "z_delta": z_delta,
        "reward": reward,
    }

