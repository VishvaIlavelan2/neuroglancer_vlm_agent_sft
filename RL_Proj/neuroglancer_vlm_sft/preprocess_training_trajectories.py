from __future__ import annotations

import argparse
import copy
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroglancer_vlm_agent.view_params import (
    apply_view_parameters,
    get_view_parameters,
)
from neuroglancer_vlm_agent.vlm_navigator.utils.nerve_visibility import (
    build_static_mask,
    classify_visibility,
    visibility_score,
)
from neuroglancer_vlm_agent.vlm_navigator.utils.reward_utils import (
    NOT_VISIBLE_REWARD,
    UNCERTAIN_REWARD_MULTIPLIER,
    VISIBLE_REWARD_MULTIPLIER,
    compute_visibility_based_reward,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "training_trajectories"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "processed_training_trajectories"
DEFAULT_CONFIG_PATH = REPO_ROOT / "neuroglancer_vlm_agent" / "config.json"
DEFAULT_STARTUP_WAIT = 5.0
DEFAULT_RENDER_WAIT = 3.0
DEFAULT_RENDER_RETRIES = 3
SEGMENT_ID_PATTERN = re.compile(r"(?P<segment_id>\d+)_data_trajectory\.json$")
STATE_KEYS_TO_COPY = (
    "dimensions",
    "position",
    "crossSectionScale",
    "projectionOrientation",
    "projectionScale",
    "layout",
    "showDefaultAnnotations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Neuroglancer training trajectories into rendered VLM inputs "
            "and ground-truth position-delta actions."
        )
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Directory containing *_data_trajectory.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for rendered images and the output manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        default="dataset_manifest.json",
        help="Filename for the consolidated output manifest inside --output-root.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to neuroglancer_vlm_agent/config.json.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=DEFAULT_STARTUP_WAIT,
        help="Seconds to wait after starting a new Neuroglancer session.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headlessly. Leave off if your local setup requires visible auth.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_state_url(url: str) -> tuple[str, dict[str, Any]]:
    if "#!" not in url:
        raise ValueError("Neuroglancer URL does not contain a state fragment.")
    base_url, fragment = url.split("#!", 1)
    state = json.loads(urllib.parse.unquote(fragment))
    return base_url, state


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def canonical_segment_id(raw_segment_id: Any) -> str | None:
    if raw_segment_id is None:
        return None
    text = str(raw_segment_id).strip()
    if not text:
        return None
    return text[1:] if text.startswith("!") else text


def get_segmentation_layer(state: dict[str, Any]) -> dict[str, Any]:
    for layer in state.get("layers", []):
        if layer.get("type") == "segmentation":
            return layer
    raise ValueError("Trajectory state does not contain a segmentation layer.")


def infer_target_segment_id(
    trajectory_path: Path,
    trajectory_steps: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    match = SEGMENT_ID_PATTERN.fullmatch(trajectory_path.name)
    segment_id_from_name = match.group("segment_id") if match else None

    first_state = trajectory_steps[0]["state"]
    segmentation_layer = get_segmentation_layer(first_state)
    positive_segments = [
        canonical_segment_id(segment_id)
        for segment_id in segmentation_layer.get("segments", [])
        if str(segment_id).strip() and not str(segment_id).startswith("!")
    ]
    positive_segments = [segment_id for segment_id in positive_segments if segment_id is not None]

    segment_query = canonical_segment_id(segmentation_layer.get("segmentQuery"))

    target_segment_id = (
        segment_id_from_name
        or (positive_segments[0] if positive_segments else None)
        or segment_query
    )
    if target_segment_id is None:
        raise ValueError(f"Unable to infer target segment id for {trajectory_path.name}")

    if positive_segments and target_segment_id not in positive_segments:
        warnings.append(
            f"{trajectory_path.name}: inferred target segment {target_segment_id} "
            f"is not the active positive segment ({positive_segments})."
        )
    if segment_query and segment_query != target_segment_id:
        warnings.append(
            f"{trajectory_path.name}: segmentQuery={segment_query} differs from "
            f"target segment {target_segment_id}; using the target segment for rendering."
        )

    return target_segment_id, warnings


def normalize_state_for_render(
    raw_state: dict[str, Any],
    target_segment_id: str,
    render_state_template: dict[str, Any],
) -> dict[str, Any]:
    state = copy.deepcopy(render_state_template)

    for key in STATE_KEYS_TO_COPY:
        if key in raw_state:
            state[key] = copy.deepcopy(raw_state[key])

    if "position" in state:
        state["position"] = [float(value) for value in state["position"]]

    apply_view_parameters(state)

    segmentation_layer = get_segmentation_layer(state)
    segmentation_layer["segments"] = [target_segment_id]
    segmentation_layer.pop("segmentQuery", None)

    selected_layer = state.get("selectedLayer")
    if isinstance(selected_layer, dict) and segmentation_layer.get("name"):
        selected_layer["layer"] = segmentation_layer["name"]

    return state


def build_state_url(state: dict[str, Any], viewer_base_url: str) -> str:
    fragment = urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    return f"{viewer_base_url}#!{fragment}"


def get_target_image_size(config: dict[str, Any]) -> tuple[int, int]:
    width = int(config.get("model_image_width", 960))
    height = int(config.get("model_image_height", 540))
    return width, height


def get_resample_lanczos(image_module: Any) -> Any:
    resampling = getattr(image_module, "Resampling", image_module)
    return getattr(resampling, "LANCZOS")


def build_ground_truth_action(
    current_position: list[float],
    next_position: list[float] | None,
) -> dict[str, Any]:
    if next_position is None:
        return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "done": True}

    return {
        "delta_x": float(next_position[0] - current_position[0]),
        "delta_y": float(next_position[1] - current_position[1]),
        "delta_z": float(next_position[2] - current_position[2]),
    }


def annotate_records_with_visibility_and_reward(
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    image_paths = [(output_root / record["image_path"]).resolve() for record in records]
    static_mask = build_static_mask(image_paths)
    consecutive_not_visible = 0

    for record, image_path in zip(records, image_paths):
        score = visibility_score(image_path, static_mask=static_mask)
        label = classify_visibility(score["dynamic_colored_fraction"])
        if label == "not_visible":
            consecutive_not_visible += 1
        else:
            consecutive_not_visible = 0

        z_delta = float(record["ground_truth_action"]["delta_z"])
        record["visibility_label"] = label
        record["consecutive_not_visible"] = consecutive_not_visible
        record["reward"] = compute_visibility_based_reward(label, z_delta)


@dataclass(frozen=True)
class RenderSettings:
    config_path: Path
    target_image_size: tuple[int, int]
    startup_wait: float
    headless: bool


class TrajectoryRenderer:
    def __init__(self, settings: RenderSettings) -> None:
        self.settings = settings
        self.render_wait = DEFAULT_RENDER_WAIT
        self.render_retries = DEFAULT_RENDER_RETRIES
        self.env = None
        self._environment_cls = None
        self._image_module = None
        self._resize_filter = None

        try:
            from ngllib import Environment
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Rendering requires ngllib and Pillow to be installed in the active Python environment."
            ) from exc

        self._environment_cls = Environment
        self._image_module = Image
        self._resize_filter = get_resample_lanczos(Image)

    def start(self, start_url: str) -> None:
        last_error: Exception | None = None
        for _ in range(self.render_retries):
            try:
                self.env = self._environment_cls(
                    headless=self.settings.headless,
                    config_path=str(self.settings.config_path),
                    verbose=False,
                    reward_function=lambda state, action, prev_state: (0.0, False),
                    start_url=start_url,
                )
                self.env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)
                time.sleep(self.settings.startup_wait)
                if not self.env.get_JSON_state():
                    raise RuntimeError("Environment returned an empty JSON state after start_session().")
                return
            except Exception as exc:
                last_error = exc
                self.close()
                time.sleep(3)

        raise RuntimeError(f"Failed to initialize Neuroglancer environment: {last_error}")

    def render_state(self, state: dict[str, Any], output_path: Path) -> None:
        if self.env is None:
            raise RuntimeError("Renderer has not been started.")

        self.env.change_JSON_state_url(state)
        time.sleep(self.render_wait)

        try:
            self.env.prev_state, self.env.prev_json = self.env.prepare_state()
        except Exception:
            pass

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.env.get_screenshot(save_path=str(output_path))

        with self._image_module.open(output_path) as captured_image:
            image = captured_image.copy()

        if image.size != self.settings.target_image_size:
            image = image.resize(self.settings.target_image_size, resample=self._resize_filter)

        image.save(output_path, format="JPEG", quality=85)

    def close(self) -> None:
        if self.env is None:
            return
        try:
            self.env.end_session()
        finally:
            self.env = None


def collect_trajectory_paths(trajectory_dir: Path) -> list[Path]:
    return sorted(trajectory_dir.glob("*_data_trajectory.json"))


def process_trajectory(
    trajectory_path: Path,
    *,
    output_root: Path,
    viewer_base_url: str,
    render_state_template: dict[str, Any],
    render_settings: RenderSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    trajectory_steps = load_json(trajectory_path)
    if not isinstance(trajectory_steps, list) or not trajectory_steps:
        raise ValueError(f"{trajectory_path.name} does not contain a non-empty list of trajectory states.")

    trajectory_path_str = str(trajectory_path.resolve())
    target_segment_id, warnings = infer_target_segment_id(trajectory_path, trajectory_steps)
    print(f"    segment_id={target_segment_id}")
    first_state = normalize_state_for_render(
        trajectory_steps[0]["state"],
        target_segment_id,
        render_state_template,
    )

    renderer = TrajectoryRenderer(render_settings)
    renderer.start(build_state_url(first_state, viewer_base_url))

    records: list[dict[str, Any]] = []
    images_dir = output_root / "images" / target_segment_id

    try:
        for step_index, step_entry in enumerate(trajectory_steps):
            current_state = normalize_state_for_render(
                step_entry["state"],
                target_segment_id,
                render_state_template,
            )
            current_position = [float(value) for value in current_state["position"]]

            next_position: list[float] | None = None
            if step_index + 1 < len(trajectory_steps):
                next_state = trajectory_steps[step_index + 1]["state"]
                next_position = [float(value) for value in next_state["position"]]

            image_path = images_dir / f"step_{step_index:04d}.jpg"
            renderer.render_state(current_state, image_path)

            records.append(
                {
                    "trajectory_id": target_segment_id,
                    "trajectory_file": trajectory_path_str,
                    "step_index": step_index,
                    "image_path": str(image_path.relative_to(output_root)).replace("\\", "/"),
                    "position": current_position,
                    "crossSectionScale": float(current_state["crossSectionScale"]),
                    "projectionOrientation": [
                        float(value) for value in current_state["projectionOrientation"]
                    ],
                    "projectionScale": float(current_state["projectionScale"]),
                    "ground_truth_action": build_ground_truth_action(
                        current_position,
                        next_position,
                    ),
                }
            )
    finally:
        renderer.close()

    annotate_records_with_visibility_and_reward(output_root, records)

    summary = {
        "trajectory_id": target_segment_id,
        "trajectory_file": trajectory_path_str,
        "viewer_base_url": viewer_base_url,
        "num_steps": len(records),
        "images_dir": str(images_dir.relative_to(output_root)).replace("\\", "/"),
    }
    return summary, records, warnings


def build_manifest_payload(
    *,
    trajectory_dir: Path,
    config_path: Path,
    target_image_size: tuple[int, int],
    trajectory_summaries: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    all_warnings: list[str],
) -> dict[str, Any]:
    return {
        "source_trajectory_dir": str(trajectory_dir),
        "config_path": str(config_path),
        "image_size": list(target_image_size),
        "view_parameters": get_view_parameters(),
        "action_format": "position_only",
        "terminal_action_policy": {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "done_included": True,
        },
        "reward_policy": {
            "type": "visibility_weighted_z_delta",
            "visible_multiplier": VISIBLE_REWARD_MULTIPLIER,
            "uncertain_multiplier": UNCERTAIN_REWARD_MULTIPLIER,
            "not_visible_reward": NOT_VISIBLE_REWARD,
        },
        "num_trajectories": len(trajectory_summaries),
        "num_records": len(all_records),
        "warnings": all_warnings,
        "trajectories": trajectory_summaries,
        "records": all_records,
    }


def main() -> None:
    args = parse_args()

    trajectory_dir = args.trajectory_dir.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config_path.resolve()
    manifest_path = output_root / args.manifest_name

    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectory directory does not exist: {trajectory_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    config = load_json(config_path)
    target_image_size = get_target_image_size(config)
    viewer_base_url, render_state_template = split_state_url(str(config["default_ngl_start_url"]))

    trajectory_paths = collect_trajectory_paths(trajectory_dir)
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectory files found in {trajectory_dir}")

    print(f"Found {len(trajectory_paths)} trajectory files in {trajectory_dir}")
    print(f"Output root: {output_root}")
    print(f"Manifest:    {manifest_path}")
    print(f"Image size:  {target_image_size[0]}x{target_image_size[1]}")
    print(f"Render wait: {DEFAULT_RENDER_WAIT:.1f}s")

    output_root.mkdir(parents=True, exist_ok=True)

    trajectory_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    render_settings = RenderSettings(
        config_path=config_path,
        target_image_size=target_image_size,
        startup_wait=args.startup_wait,
        headless=args.headless,
    )

    for index, trajectory_path in enumerate(trajectory_paths, start=1):
        print(f"[{index}/{len(trajectory_paths)}] Processing {trajectory_path.name}")
        summary, records, warnings = process_trajectory(
            trajectory_path,
            output_root=output_root,
            viewer_base_url=viewer_base_url,
            render_state_template=render_state_template,
            render_settings=render_settings,
        )

        trajectory_summaries.append(summary)
        all_records.extend(records)
        all_warnings.extend(warnings)

        manifest_payload = build_manifest_payload(
            trajectory_dir=trajectory_dir,
            config_path=config_path,
            target_image_size=target_image_size,
            trajectory_summaries=trajectory_summaries,
            all_records=all_records,
            all_warnings=all_warnings,
        )
        write_json(manifest_path, manifest_payload)

    print(f"Wrote manifest with {len(all_records)} records to {manifest_path}")
    if all_warnings:
        print("Warnings:")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
