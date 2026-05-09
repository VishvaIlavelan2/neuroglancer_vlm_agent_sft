from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from neuroglancer_vlm_agent.vlm_navigator.utils.action_quantization import (
        quantize_action,
        quantize_previous_outcome,
    )
except ModuleNotFoundError:
    from vlm_navigator.utils.action_quantization import (
        quantize_action,
        quantize_previous_outcome,
    )

SYSTEM_PROMPT = """You  are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer to find the highest-Z tip of a specific neuron segment.

  PANELS:
  - LEFT: 2D cross-section of current Z slice. Grey = raw EM tissue. Colored overlay = your target neuron segment. Crosshair = current position.
  - RIGHT: 3D mesh projection of the neuron so far.

  YOUR TASK: Reach the axon terminus — the highest Z position where the colored segment still exists. You are done when the segment disappears and cannot be recovered by x/y search.

  STRATEGY:
  1. LOOK at the screenshot. Is colored neuron tissue visible?
  2. If the neuron is clearly visible and roughly centered: advance with delta_z = 50-150. Only use smaller z steps (10-30) if the neuron is near the edge of the frame or looks like it's about to exit.
  3. If the neuron is visible but off-center: make a BOLD x/y correction (250-1000 units)
  4. If NO visible neuron (black/empty): you overshot or drifted. DECREASE delta_z (go backward) or adjust x/y to find the neuron again
  5. If the neuron shifts left/right/up/down between steps, compensate with x/y deltas
  6. Always consult the 3D mesh (right panel) to determine where the nerve continues in 3D space, and use this to guide your x/y positioning.
  
  SCALE REFERENCE:
  - The scale bar in the bottom-left of the 2D panel is approximately 1/8th of the panel width.
  - x/y steps of 10-20 are almost invisible. Use 50-200 for meaningful repositioning.
  - z steps of 5-10 are wastefully small when the neuron is clearly visible. Default to 50+.

  COORDINATE SYSTEM:
  - Y-axis points DOWN — increasing delta_y moves the view downward on screen.
  - To move toward something above centre, use a negative delta_y.
  - To move toward something below centre, use a positive delta_y.
  
  REWARD INTERPRETATION:
  - Positive reward means the previous move was good: it increased Z while staying on the neuron.
  - Smaller positive reward means some progress, but with weaker confidence or visibility.
  - Negative reward means the previous move was bad: you likely lost the neuron or made unhelpful progress.
  - Use the previous outcome summary as feedback: if reward was positive, continue a similar strategy; if reward was negative, correct course.

  IMPORTANT:
  - The neuron curves through 3D space — it won't stay at the same x,y as you change z
  - Small z steps (10-30) are safer than large ones (50-100) when the neuron is near an edge
  - Re-centering the neuron (x/y adjustment) is MORE important than z progress
  - You may end the run early if you are confident further z-progress is impossible
    (e.g. the nerve has been lost for multiple steps and x/y corrections have failed).
    If ending, include "done": true in the same JSON object alongside the deltas.

  Respond with ONLY a JSON object.
  {"delta_x": 0, "delta_y": 0, "delta_z": 50}
"""

REFERENCE_PROMPT_HORIZON = 120


STEP_PROMPT_TEMPLATE = (
    "Reference horizon remaining: {reference_horizon_remaining}. "
    "Visibility heuristic this step: {visibility_label}. "
    "Consecutive not-visible frames: {consecutive_not_visible}. "
    "Z progress from start: {z_from_start:+.1f}. "
    "Previous action: {previous_action_summary}. "
    "Reward from previous action: {previous_outcome_summary}. "
    "Screenshot has 2D and 3D views. "
    "Return ONLY the next action as JSON."
)


def format_previous_action_summary(previous_action: Mapping[str, Any] | None) -> str:
    """Summarize the previous action in a compact, deployment-safe form."""
    if previous_action is None:
        return "none (first step)"

    quantized_action = quantize_action(dict(previous_action))
    dx = float(quantized_action.get("delta_x", 0.0))
    dy = float(quantized_action.get("delta_y", 0.0))
    dz = float(quantized_action.get("delta_z", 0.0))
    summary = f"dx={dx:+.1f}, dy={dy:+.1f}, dz={dz:+.1f}"
    if quantized_action.get("done"):
        summary += ", done=true"
    return summary


def format_previous_outcome_summary(previous_outcome: Mapping[str, Any] | None) -> str:
    """Summarize the previous step outcome in a compact, deployment-safe form."""
    if previous_outcome is None:
        return "none (first step)"

    quantized_outcome = quantize_previous_outcome(dict(previous_outcome))
    reward = float(quantized_outcome.get("reward", 0.0))
    z_delta = float(quantized_outcome.get("z_delta", 0.0))
    visibility_label = str(
        quantized_outcome.get("visibility_label", quantized_outcome.get("label", "unknown"))
    )
    if reward > 0:
        quality = "good"
    elif reward < 0:
        quality = "bad"
    else:
        quality = "neutral"
    return (
        f"reward={reward:+.1f}, quality={quality}, "
        f"z_delta={z_delta:+.1f}, visibility={visibility_label}"
    )


def build_step_prompt(
    *,
    step: int,
    z_from_start: float,
    previous_action: Mapping[str, Any] | None,
    previous_outcome: Mapping[str, Any] | None = None,
    visibility_label: str = "unknown",
    consecutive_not_visible: int = 0,
    reference_prompt_horizon: int = REFERENCE_PROMPT_HORIZON,
    step_prompt_template: str | None = None,
) -> str:
    """Build the runtime/training step prompt from compact structured state."""
    template = STEP_PROMPT_TEMPLATE if step_prompt_template is None else step_prompt_template
    reference_horizon_remaining = max(1, int(reference_prompt_horizon) - int(step) + 1)
    return template.format(
        reference_horizon_remaining=reference_horizon_remaining,
        visibility_label=visibility_label,
        consecutive_not_visible=int(consecutive_not_visible),
        z_from_start=z_from_start,
        previous_action_summary=format_previous_action_summary(previous_action),
        previous_outcome_summary=format_previous_outcome_summary(previous_outcome),
    )
