from __future__ import annotations

VISIBLE_REWARD_MULTIPLIER = 1.0
UNCERTAIN_REWARD_MULTIPLIER = 0.5
NOT_VISIBLE_REWARD = -1.0


def compute_visibility_based_reward(visibility_label: str, z_delta: float) -> float:
    z_delta = float(z_delta)
    if visibility_label == "visible":
        return VISIBLE_REWARD_MULTIPLIER * z_delta
    if visibility_label == "uncertain":
        return UNCERTAIN_REWARD_MULTIPLIER * z_delta
    return NOT_VISIBLE_REWARD
