#!/usr/bin/env python3
"""Generate a Spatial Annotation CSV for ProcTHOR dual-agent benchmarks.

Columns align with the AI2-THOR dual-agent CSV schema:
    Task ID, Completed, golden_action, instruction, token_total, failure_reason,
    actual_actions, golden_actions_count, actual_actions_count

Usage:
    python dual_agent/scripts/generate_procthor_csv.py \
        --task-dir dual_agent/task_mutil_procthor \
        --output "experiments/csv/procthor/dual/Spatial-Annotation-procthor.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


COLUMNS = [
    "Task ID",
    "Completed",
    "golden_action",
    "instruction",
    "token_total",
    "failure_reason",
    "actual_actions",
    "golden_actions_count",
    "actual_actions_count",
]


def extract_golden_info(task_json: Path) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Return (golden_action_text, golden_action_count, instruction) from task.json."""
    try:
        with open(task_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None

    instruction = data.get("instruction") or data.get("target_description") or data.get("task_name")
    golden = data.get("golden_actions")
    actions: list[str] = []
    steps_int: Optional[int] = None
    if isinstance(golden, dict):
        s = golden.get("steps")
        if isinstance(s, int):
            steps_int = int(s)
        raw_actions = golden.get("actions") or []
        if isinstance(raw_actions, list):
            actions = [str(a).strip() for a in raw_actions if str(a).strip()]
    elif isinstance(golden, list):
        actions = [str(a).strip() for a in golden if str(a).strip()]
    elif isinstance(golden, str):
        actions = [a.strip() for a in golden.split(",") if a.strip()]

    golden_action_text = " | ".join(actions) if actions else None
    if steps_int is not None:
        golden_action_count = steps_int
    elif actions:
        non_done = [a for a in actions if a.upper() != "DONE"]
        golden_action_count = len(non_done) if non_done else len(actions)
    else:
        golden_action_count = None

    return golden_action_text, golden_action_count, instruction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=str,
        default="dual_agent/task_mutil_procthor",
        help="Task folder root (contains procthor<xxx>/task.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Task IDs to exclude (e.g. stuck cases).",
    )
    args = parser.parse_args()

    task_root = Path(args.task_dir)
    if not task_root.exists():
        print(f"ERROR: task dir not found: {task_root}")
        return 1

    exclude_set = set(args.exclude or [])
    task_dirs = sorted(d for d in task_root.iterdir() if d.is_dir() and (d / "task.json").exists())
    rows: list[list[str]] = []

    for task_dir in task_dirs:
        task_id = task_dir.name
        if task_id in exclude_set:
            print(f"  • Skipping excluded task: {task_id}")
            continue

        golden_text, golden_count, instruction = extract_golden_info(task_dir / "task.json")
        row = [
            task_id,
            "",
            golden_text or "",
            instruction or "",
            "",
            "",
            "",
            str(golden_count) if golden_count is not None else "",
            "",
        ]
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"✓ Generated {len(rows)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
