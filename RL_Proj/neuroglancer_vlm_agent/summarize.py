"""
Aggregate all log.json files under results/manual_test/ into a CSV.

Produces two files in results/:
  summary.csv      - one row per run
  steps.csv        - one row per step across all runs
"""

import argparse
import csv
import json
import re
from pathlib import Path


VISIBLE_GRAPH_LABELS = {"visible", "uncertain"}


def parse_folder_name(folder_name: str) -> dict:
    """Best-effort parse of model/position/trial from folder name."""
    match = re.match(r"^(.+?)_pos(\d+)_trial(\d+)$", folder_name)
    if match:
        return {"model": match.group(1), "position_id": int(match.group(2)), "trial": int(match.group(3))}
    match = re.match(r"^(.+?)_pos(\d+)$", folder_name)
    if match:
        return {"model": match.group(1), "position_id": int(match.group(2)), "trial": 1}
    return {"model": folder_name, "position_id": None, "trial": 1}


def compute_visibility_filtered_z_series(steps: list[dict], start_z: float) -> list[dict]:
    """Reconstruct graph Z using only visible/uncertain updates."""
    graph_z = float(start_z)
    rows = []
    for step in steps:
        label = step.get("nerve_visible")
        if label in VISIBLE_GRAPH_LABELS:
            graph_z += float(step.get("z_delta", 0.0))
        rows.append({"graph_z": graph_z, "label": label})
    return rows


def collect_runs(results_dir: Path) -> list[dict]:
    runs = []
    for log_path in sorted(results_dir.rglob("log.json")):
        folder = log_path.parent
        with open(log_path) as handle:
            steps = json.load(handle)
        if not steps:
            continue

        meta = parse_folder_name(folder.name)
        start_z = steps[0]["position_before"][2]
        graph_rows = compute_visibility_filtered_z_series(steps, start_z)
        final_z = graph_rows[-1]["graph_z"]
        early_stop = any(step.get("early_stop") for step in steps)
        agent_stop = steps[-1].get("action") is not None and bool(
            (steps[-1].get("action") or {}).get("done", False)
        )

        nerve_counts = {"visible": 0, "uncertain": 0, "not_visible": 0}
        best_z_on_nerve = start_z
        for step, graph_row in zip(steps, graph_rows):
            label = step.get("nerve_visible")
            if label in nerve_counts:
                nerve_counts[label] += 1
            if label in ("visible", "uncertain"):
                z_value = graph_row["graph_z"]
                if z_value > best_z_on_nerve:
                    best_z_on_nerve = z_value

        runs.append({
            "folder": str(folder.relative_to(results_dir.parent)),
            "model": meta["model"],
            "position_id": meta["position_id"],
            "trial": meta["trial"],
            "steps_taken": len(steps),
            "start_z": start_z,
            "final_z": final_z,
            "z_gained": final_z - start_z,
            "best_z_on_nerve": best_z_on_nerve,
            "best_z_on_nerve_gained": best_z_on_nerve - start_z,
            "early_stop_nerve": early_stop,
            "agent_stop": agent_stop,
            "steps_visible": nerve_counts["visible"],
            "steps_uncertain": nerve_counts["uncertain"],
            "steps_not_visible": nerve_counts["not_visible"],
        })
    return runs


def collect_steps(results_dir: Path, runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        log_path = results_dir.parent / run["folder"] / "log.json"
        if not log_path.exists():
            continue
        with open(log_path) as handle:
            steps = json.load(handle)
        graph_rows = compute_visibility_filtered_z_series(steps, run["start_z"])
        for step, graph_row in zip(steps, graph_rows):
            rows.append({
                "folder": run["folder"],
                "model": run["model"],
                "position_id": run["position_id"],
                "trial": run["trial"],
                "step": step["step"],
                "z": graph_row["graph_z"],
                "z_delta": step.get("z_delta", 0),
                "nerve_visible": step.get("nerve_visible", ""),
                "early_stop": step.get("early_stop", False),
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"  No data for {path.name}, skipping.")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results/manual_test",
        help="Root directory to scan for log.json files (default: results/manual_test)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        raise SystemExit(1)

    out_dir = Path("results")
    runs = collect_runs(results_dir)
    steps = collect_steps(results_dir, runs)

    print(f"Found {len(runs)} runs, {len(steps)} total steps.")
    write_csv(out_dir / "summary.csv", runs)
    write_csv(out_dir / "steps.csv", steps)
