# fine tuning done with tinker.
# essentially only implemented functionality corresponds to trajectory collection / data formatting
# all finetuning logic handled by tinker

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuroglancer_vlm_agent.shared_prompts import (
    REFERENCE_PROMPT_HORIZON,
    build_step_prompt as build_shared_step_prompt,
)
from neuroglancer_vlm_agent.vlm_navigator.utils.action_quantization import (
    ACTION_QUANTIZATION,
    quantize_action,
)
from neuroglancer_vlm_agent.vlm_navigator.utils.reward_utils import (
    compute_visibility_based_reward,
)

from .prompts import load_runtime_prompts


DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "processed_training_trajectories" / "dataset_manifest.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "neuroglancer_tinker_sft_conversations.jsonl"
DEFAULT_INFO_PATH = Path(__file__).resolve().parent / "data" / "neuroglancer_tinker_sft_dataset_info.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def action_to_text(action: dict[str, Any]) -> str:
    quantized_action = quantize_action(action)
    payload: dict[str, Any] = {
        "delta_x": quantized_action["delta_x"],
        "delta_y": quantized_action["delta_y"],
        "delta_z": quantized_action["delta_z"],
    }
    if quantized_action.get("done"):
        payload["done"] = True
    return json.dumps(payload)


def should_include_training_example(action: dict[str, Any]) -> bool:
    return not bool(action.get("done", False))


def format_step_prompt(
    step_prompt_template: str,
    *,
    step: int,
    z_from_start: float,
    previous_action: dict[str, Any] | None,
    previous_outcome: dict[str, Any] | None,
    visibility_label: str,
    consecutive_not_visible: int,
) -> str:
    return build_shared_step_prompt(
        step=step,
        z_from_start=z_from_start,
        previous_action=previous_action,
        previous_outcome=previous_outcome,
        visibility_label=visibility_label,
        consecutive_not_visible=consecutive_not_visible,
        step_prompt_template=step_prompt_template,
    )


def group_records_by_trajectory(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["trajectory_id"], []).append(record)
    for trajectory_records in grouped.values():
        trajectory_records.sort(key=lambda record: record["step_index"])
    return grouped


def build_step_states(
    manifest_root: Path,
    trajectory_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if trajectory_records and all(
        "visibility_label" in record and "consecutive_not_visible" in record
        for record in trajectory_records
    ):
        return [
            {
                "image": str((manifest_root / record["image_path"]).resolve()),
                "label": str(record["visibility_label"]),
                "visibility_label": str(record["visibility_label"]),
                "consecutive_not_visible": int(record["consecutive_not_visible"]),
                "z_delta": float(record["ground_truth_action"]["delta_z"]),
                "reward": float(
                    record.get(
                        "reward",
                        compute_visibility_based_reward(
                            str(record["visibility_label"]),
                            float(record["ground_truth_action"]["delta_z"]),
                        ),
                    )
                ),
            }
            for record in trajectory_records
        ]

    image_paths = [
        (manifest_root / record["image_path"]).resolve()
        for record in trajectory_records
    ]
    from neuroglancer_vlm_agent.vlm_navigator.utils.nerve_visibility import (
        build_static_mask,
        classify_visibility,
        visibility_score,
    )

    static_mask = build_static_mask(image_paths)
    states: list[dict[str, Any]] = []
    consecutive_not_visible = 0

    for image_path, record in zip(image_paths, trajectory_records):
        score = visibility_score(image_path, static_mask=static_mask)
        label = classify_visibility(score["dynamic_colored_fraction"])
        if label == "not_visible":
            consecutive_not_visible += 1
        else:
            consecutive_not_visible = 0
        z_delta = float(record["ground_truth_action"]["delta_z"])
        states.append(
            {
                "image": str(image_path),
                "label": label,
                "visibility_label": label,
                "consecutive_not_visible": consecutive_not_visible,
                "z_delta": z_delta,
                "reward": compute_visibility_based_reward(label, z_delta),
                **score,
            }
        )

    if len(states) != len(trajectory_records):
        raise ValueError(
            f"Expected {len(trajectory_records)} visibility states, got {len(states)}"
        )
    return states


def build_conversations(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    manifest_root = manifest_path.resolve().parent
    prompt_bundle = load_runtime_prompts()
    trajectories = manifest["trajectories"]
    records_by_trajectory = group_records_by_trajectory(manifest["records"])

    conversations: list[dict[str, Any]] = []
    excluded_done_examples = 0

    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        trajectory_records = records_by_trajectory[trajectory_id]
        step_states = build_step_states(manifest_root, trajectory_records)
        start_z = float(trajectory_records[0]["position"][2])

        for record_index, record in enumerate(trajectory_records):
            if not should_include_training_example(record["ground_truth_action"]):
                excluded_done_examples += 1
                continue

            step_state = step_states[record_index]
            step_text = format_step_prompt(
                prompt_bundle.step_prompt_template,
                step=record["step_index"] + 1,
                z_from_start=float(record["position"][2]) - start_z,
                previous_action=(
                    None
                    if record_index == 0
                    else trajectory_records[record_index - 1]["ground_truth_action"]
                ),
                previous_outcome=(
                    None
                    if record_index == 0
                    else step_states[record_index - 1]
                ),
                visibility_label=step_state["label"],
                consecutive_not_visible=step_state["consecutive_not_visible"],
            )

            conversations.append(
                {
                    "example_id": f"{trajectory_id}_step_{record['step_index']:04d}",
                    "messages": [
                        {"role": "system", "content": prompt_bundle.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": step_text},
                                {
                                    "type": "image",
                                    "image": str((manifest_root / record["image_path"]).resolve()),
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": action_to_text(record["ground_truth_action"]),
                        },
                    ],
                }
            )

    info = {
        "manifest_path": str(manifest_path.resolve()),
        "prompt_source": str(prompt_bundle.source_path),
        "system_prompt": prompt_bundle.system_prompt,
        "step_prompt_template": prompt_bundle.step_prompt_template,
        "action_quantization": ACTION_QUANTIZATION,
        "excluded_done_examples": excluded_done_examples,
        "view_parameters": manifest.get("view_parameters"),
        "reference_prompt_horizon": REFERENCE_PROMPT_HORIZON,
        "num_examples": len(conversations),
        "num_trajectories": len(trajectories),
    }
    return conversations, info
