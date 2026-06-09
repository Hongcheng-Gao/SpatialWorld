#!/usr/bin/env python3
"""Detect ProcTHOR tasks where the agent cannot move at all after initialization.

Procedure per task:
  1. Load task.json (from mllm_base_agent/dual_agent/procthor/task_mutil_procthor/<task_id>).
  2. Launch ProcTHOR env with the task's scene_index and init_actions.
  3. Try each direction (MoveAhead/Back/Left/Right) at grid_size step.
  4. If all four moves fail, the task is considered "stuck" — both agents
     would spawn at the same unreachable pose, so the task is not runnable.
  5. Close env; record stuck task IDs.

Optional: pass --rewrite-csv <CSV> to rebuild the CSV without stuck rows
          (keeps other rows as-is, preserves Completed values).

Usage:
    python dual_agent/scripts/filter_stuck_tasks.py \\
        --task-dir mllm_base_agent/dual_agent/procthor/task_mutil_procthor \\
        --config dual_agent/config_close_Gemini-2.5-pro.yaml \\
        --headless \\
        --rewrite-csv "experiments/csv/procthor/dual/Spatial-Annotation-procthor.csv" \\
        --output dual_agent/stuck_tasks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
DUAL_AGENT_DIR = _THIS.parents[1]
TASK_ROOT = DUAL_AGENT_DIR / "task_mutil_procthor"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action_parser import parse_action_string  # noqa: E402
from envs.procthor_wrapper import ProcTHOREnvWrapper  # noqa: E402
from scripts.evaluate_actions_procthor import load_init_actions_for_task  # noqa: E402


MOVE_ACTIONS = ["MoveAhead(0.25)", "MoveBack(0.25)", "MoveLeft(0.25)", "MoveRight(0.25)"]


def load_yaml_config(config_path: Path, headless: bool) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if headless:
        data.setdefault("env", {})
        import os

        display = os.environ.get("DISPLAY", "").strip()
        if display:
            data["env"]["x_display"] = display
            if isinstance(data["env"].get("platform"), str) and data["env"]["platform"].lower() == "cloudrendering":
                data["env"]["platform"] = None
        else:
            data["env"]["platform"] = "CloudRendering"
    return data


def load_task_json(task_dir: Path) -> Optional[Dict]:
    task_json = task_dir / "task.json"
    if not task_json.exists():
        return None
    try:
        with open(task_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_moveability_check(task_id: str, task_config: Dict, base_config: Dict, headless: bool) -> Dict:
    """Return {'stuck': bool, 'success_actions': [...], 'error': str|None}."""
    out = {"task_id": task_id, "stuck": False, "success_actions": [], "error": None}
    task_folder = TASK_ROOT / task_id
    init_actions = load_init_actions_for_task(str(task_folder / "task.json"))

    cfg = deepcopy(base_config)
    cfg["task"] = task_config
    cfg["init_actions"] = init_actions or []

    env = None
    try:
        env = ProcTHOREnvWrapper(
            scene_index=task_config.get("scene_index", 0),
            output_dir=str(PROJECT_ROOT / "dual_agent" / "filter_outputs" / task_id),
            config=cfg,
            headless=headless,
        )
        instruction = task_config.get("instruction") or task_config.get("target_description") or ""
        env.reset(instruction)

        successes: List[str] = []
        for action_str in MOVE_ACTIONS:
            try:
                action_dict = parse_action_string(action_str)
                _, error_msg = env.step_with_action_dict(action_dict)
                if not error_msg:
                    successes.append(action_str)
            except Exception as e:
                out["error"] = f"{action_str}: {e}"
        out["success_actions"] = successes
        out["stuck"] = len(successes) == 0
    except Exception as e:
        out["error"] = f"{e}\n{traceback.format_exc()}"
        out["stuck"] = True  # unusable for benchmark anyway
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return out


def rewrite_csv_without_stuck(csv_path: Path, stuck_ids: List[str]) -> None:
    if not csv_path.exists():
        print(f"⚠️  CSV not found; skip rewrite: {csv_path}")
        return
    stuck_set = set(stuck_ids)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    kept = [header]
    removed = 0
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        tid = row[0].strip()
        if tid in stuck_set:
            removed += 1
            continue
        kept.append(row)
    backup = csv_path.with_suffix(csv_path.suffix + ".prefilter.backup")
    if not backup.exists():
        import shutil as _shutil

        _shutil.copy2(csv_path, backup)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept)
    print(f"✓ CSV rewritten: removed {removed} stuck rows (backup: {backup.name})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=str, default=str(TASK_ROOT))
    parser.add_argument("--config", type=str, required=True, help="Base yaml config (env section is used).")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=str, default="dual_agent/stuck_tasks.json", help="Where to write the stuck-tasks report.")
    parser.add_argument("--rewrite-csv", type=str, default=None, help="If set, rewrite this CSV removing stuck rows.")
    parser.add_argument("--only", type=str, nargs="+", default=None, help="Optional subset of task IDs to check.")
    args = parser.parse_args()

    task_root = Path(args.task_dir)
    if not task_root.exists():
        print(f"❌ Task dir not found: {task_root}")
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    base_config = load_yaml_config(config_path, headless=args.headless)

    subset = set(args.only or [])
    task_dirs = sorted(d for d in task_root.iterdir() if d.is_dir() and (d / "task.json").exists())
    if subset:
        task_dirs = [d for d in task_dirs if d.name in subset]

    reports: List[Dict] = []
    stuck_ids: List[str] = []
    for idx, task_dir in enumerate(task_dirs, 1):
        print(f"\n[{idx}/{len(task_dirs)}] Checking {task_dir.name}...")
        task_json = load_task_json(task_dir)
        if task_json is None:
            print(f"  ⚠️  cannot read task.json, skipping")
            continue
        task_json.setdefault("task_folder_path", str(task_dir.resolve()))
        task_json.setdefault("name", task_json.get("task_id") or task_dir.name)
        report = run_moveability_check(task_dir.name, task_json, base_config, headless=args.headless)
        if report.get("error"):
            print(f"  ⚠️  error: {report['error'][:200]}")
        print(f"  ✓ successes: {report['success_actions']}  stuck={report['stuck']}")
        reports.append(report)
        if report["stuck"]:
            stuck_ids.append(task_dir.name)

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_dir": str(task_root),
                "config": str(config_path),
                "total": len(reports),
                "stuck_count": len(stuck_ids),
                "stuck_task_ids": stuck_ids,
                "reports": reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✓ Report written: {report_path}")
    print(f"Total checked: {len(reports)}  stuck: {len(stuck_ids)}")
    if stuck_ids:
        print(f"Stuck: {stuck_ids}")

    if args.rewrite_csv:
        rewrite_csv_without_stuck(Path(args.rewrite_csv), stuck_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
