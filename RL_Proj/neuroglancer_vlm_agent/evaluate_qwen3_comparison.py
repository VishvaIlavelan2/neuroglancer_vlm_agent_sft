"""
Run the standard Neuroglancer evaluation loop for the hardcoded base and
fine-tuned Qwen3-VL-235B models.
"""

from __future__ import annotations

import argparse
import json

from run_manual_test import MODEL_PRESETS, SEGMENTS, run_manual_test


COMPARISON_MODELS = ("qwen3-vl-235b-base", "qwen3-vl-235b-ft")


def _resolve_positions(raw_positions: list[str] | None) -> list[int]:
    with open("vlm_navigator/config/starting_positions.json") as handle:
        all_positions = [p["id"] for p in json.load(handle)]

    if raw_positions is None or raw_positions == ["all"]:
        return all_positions
    return [int(position_id) for position_id in raw_positions]


def _resolve_segment(raw_segment: str | None) -> str | None:
    if raw_segment is None:
        return None
    label_to_id = {label: segment_id for segment_id, label in SEGMENTS.items()}
    return label_to_id.get(raw_segment, raw_segment)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the base and fine-tuned Qwen3-VL-235B models with the "
            "standard Neuroglancer navigation loop."
        )
    )
    parser.add_argument(
        "--position",
        nargs="+",
        metavar="POS",
        default=["all"],
        help="One or more starting position IDs, or 'all' (default).",
    )
    parser.add_argument("--steps", type=int, default=14, help="Max steps per episode")
    parser.add_argument("--trials", type=int, default=1, help="Sequential trials per position")
    parser.add_argument(
        "--stop-mode",
        default="agent",
        choices=["fixed", "agent"],
        help="fixed=run full step budget, agent=allow done=true early stop",
    )
    parser.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=8,
        help="Minimum steps before accepting an agent stop request",
    )
    parser.add_argument(
        "--post-step-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after each env.step() for tiles to settle",
    )
    parser.add_argument(
        "--segment",
        default=None,
        choices=list(SEGMENTS.keys()) + list(SEGMENTS.values()),
        metavar="SEG",
        help="Optional segment ID or segment label override",
    )
    parser.add_argument("--debug", action="store_true", help="Save per-step prompts and responses")
    args = parser.parse_args()

    missing_models = [name for name in COMPARISON_MODELS if name not in MODEL_PRESETS]
    if missing_models:
        raise RuntimeError(f"Missing hardcoded model preset(s): {missing_models}")

    position_ids = _resolve_positions(args.position)
    resolved_segment = _resolve_segment(args.segment)

    total = len(COMPARISON_MODELS) * len(position_ids) * args.trials
    completed = 0
    failed = 0

    for position_id in position_ids:
        for trial in range(1, args.trials + 1):
            for model_name in COMPARISON_MODELS:
                completed += 1
                print(
                    f"\n[Comparison {completed}/{total}] "
                    f"model={model_name} position={position_id} trial={trial}"
                )
                try:
                    run_manual_test(
                        model_name=model_name,
                        position_id=position_id,
                        max_steps=args.steps,
                        save_debug=args.debug,
                        stop_mode=args.stop_mode,
                        trial=trial,
                        min_steps_before_stop=args.min_steps_before_stop,
                        post_step_delay=args.post_step_delay,
                        segment_id=resolved_segment,
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        f"\n  [ERROR] model={model_name} position={position_id} "
                        f"trial={trial} failed: {exc}"
                    )
                    print("  Continuing with remaining runs...\n")

    if failed:
        print(f"{failed}/{total} run(s) failed.")


if __name__ == "__main__":
    main()
