"""Analyze game benchmark outputs and export a combined CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GAME_ORDER = ["block3d", "maze", "maze3d_pro", "rubik", "snake"]
STANDARD_TASK_ID = re.compile(
    r"^(block3d|maze|maze3d_pro|rubik)/level_\d{2}$|^(snake)/run_\d{3}$"
)
MOVEMENT_ACTIONS = {"move_forward", "move_backward"}

CSV_EXPORT_CONFIGS = [
    {
        "game": "block3d",
        "subdir_pattern": "level_*",
        "summary_pattern": "block3d_evaluation_summary_*.json",
        "response_pattern": "block3d_model_responses_*.json",
        "completed_key": "won",
        "fields": [
            "level",
            "level_name",
            "won",
            "matched_blocks",
            "target_blocks",
            "completion_percentage",
            "steps_taken",
            "max_steps",
            "score",
            "game_over",
            "total_time",
            "total_frames",
            "model",
        ],
    },
    {
        "game": "maze",
        "subdir_pattern": "level_*",
        "summary_pattern": "maze_evaluation_summary_*.json",
        "response_pattern": "maze_model_responses_*.json",
        "completed_key": "success",
        "fields": [
            "level_number",
            "maze_file",
            "success",
            "steps_taken",
            "max_steps",
            "distance_to_exit",
            "total_time",
            "total_frames",
            "model",
        ],
    },
    {
        "game": "maze3d_pro",
        "subdir_pattern": "level_*",
        "summary_pattern": "maze3d_pro_evaluation_summary_*.json",
        "response_pattern": "maze3d_pro_model_responses_*.json",
        "completed_key": "success",
        "fields": [
            "level_number",
            "maze_file",
            "success",
            "steps_taken",
            "max_steps",
            "distance_to_exit",
            "current_floor",
            "total_floors",
            "total_time",
            "total_frames",
            "model",
        ],
    },
    {
        "game": "rubik",
        "subdir_pattern": "level_*",
        "summary_pattern": "rubik_evaluation_summary_*.json",
        "response_pattern": "rubik_model_responses_*.json",
        "completed_key": "cube_solved",
        "fields": [
            "config_name",
            "cube_solved",
            "steps",
            "max_steps",
            "time_elapsed",
            "total_time",
            "total_frames",
            "model",
        ],
    },
    {
        "game": "snake",
        "subdir_pattern": "run_*",
        "summary_pattern": "snake_evaluation_summary_*.json",
        "response_pattern": "snake_model_responses_*.json",
        "completed_key": "game_over",
        "fields": [
            "score",
            "snake_length",
            "steps_taken",
            "max_steps",
            "game_over",
            "early_stopped",
            "early_stop_score",
            "distance_to_food",
            "total_time",
            "total_frames",
            "model",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze game benchmark outputs.")
    parser.add_argument(
        "path",
        nargs="?",
        default="logs",
        help="Benchmark output directory.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Output CSV path. Default: <path>/results_summary.csv.",
    )
    parser.add_argument(
        "--no-merge-csv",
        action="store_true",
        help="Do not merge existing rows from the target CSV.",
    )
    return parser.parse_args()


ARGS = parse_args()
BASE_DIR = Path(ARGS.path)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def is_standard_task_dir(game: str, dirname: str) -> bool:
    if game == "snake":
        return bool(re.fullmatch(r"run_\d{3}", dirname))
    return bool(re.fullmatch(r"level_\d{2}", dirname))


def task_dirs_for(cfg: Dict[str, Any]) -> List[Path]:
    game_dir = BASE_DIR / cfg["game"]
    if not game_dir.exists():
        return []
    return sorted(
        d
        for d in game_dir.glob(cfg["subdir_pattern"])
        if d.is_dir() and is_standard_task_dir(cfg["game"], d.name)
    )


def csv_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    game = row.get("Game", "")
    task = row.get("Task ID", "").split("/", 1)[-1]
    return (GAME_ORDER.index(game) if game in GAME_ORDER else 99, task)


def load_existing_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return [
            row
            for row in csv.DictReader(f)
            if STANDARD_TASK_ID.match(row.get("Task ID", ""))
        ]


def merge_csv_fieldnames(*row_groups: Iterable[Dict[str, Any]]) -> List[str]:
    fieldnames = ["Game", "Task ID", "Completed", "actual_actions", "actual_actions_count", "actions"]
    seen = set(fieldnames)
    for rows in row_groups:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    return fieldnames


def normalize_csv_row(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fieldnames}


def fmt_count(value: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{value}/{total} ({100 * value / total:.1f}%)"


def check_missing(dirs: Iterable[Path], pattern: str) -> List[str]:
    return [d.name for d in dirs if not list(d.glob(pattern))]


def collect_all_missing() -> List[Tuple[str, List[str]]]:
    incomplete = []
    for cfg in CSV_EXPORT_CONFIGS:
        dirs = task_dirs_for(cfg)
        if not dirs:
            continue
        missing = check_missing(dirs, cfg["summary_pattern"])
        if missing:
            incomplete.append((cfg["game"], missing))
    return incomplete


def report_missing(game: str, missing: List[str]) -> None:
    print(f"\n[{game}] missing summary files; skipped analysis for these tasks:")
    for name in missing:
        print(f"  - {name}")


def extract_timestamp(name: str) -> Optional[str]:
    match = re.search(r"(\d{8}_\d{6})(?=\.json$)", name)
    return match.group(1) if match else None


def summary_step_count(summary: Dict[str, Any]) -> int:
    for key in ("steps_taken", "steps"):
        value = summary.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 10**9


def find_export_summary_file(task_dir: Path, cfg: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    files = sorted(task_dir.glob(cfg["summary_pattern"]))
    if not files:
        return None, None

    if cfg["game"] != "maze3d_pro":
        summary_file = files[-1]
        return summary_file, load_json(summary_file)

    loaded = [(idx, path, load_json(path)) for idx, path in enumerate(files)]
    _, summary_file, summary = min(
        loaded,
        key=lambda item: (
            not bool(item[2].get(cfg["completed_key"])),
            summary_step_count(item[2]),
            -item[0],
        ),
    )
    return summary_file, summary


def find_matching_response_file(task_dir: Path, pattern: str, summary_file: Path) -> Optional[Path]:
    response_files = sorted(task_dir.glob(pattern))
    if not response_files:
        return None

    timestamp = extract_timestamp(summary_file.name)
    if timestamp:
        matched = [path for path in response_files if timestamp in path.name]
        if matched:
            return matched[-1]
    return response_files[-1]


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}

    text = value.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_granularity(value: Any) -> Optional[str]:
    if value is None:
        return None
    granularity = str(value).strip().lower()
    if granularity == "midium":
        granularity = "medium"
    return granularity if granularity in {"small", "medium", "large"} else None


def response_action_arguments(item: Dict[str, Any], action: str) -> Dict[str, Any]:
    fallback = {}
    for call in item.get("function_calls") or []:
        args = parse_json_object(call.get("arguments", {}))
        if not fallback:
            fallback = args
        if call.get("function_name") == action:
            return args
    return fallback or parse_json_object(item.get("response_text", ""))


def format_action(item: Dict[str, Any]) -> Optional[str]:
    action = item.get("action_taken")
    if action is None:
        return None

    action = str(action).strip()
    if not action:
        return None

    base_action = action.removeprefix("default_")
    if base_action in MOVEMENT_ACTIONS:
        args = response_action_arguments(item, base_action)
        granularity = normalize_granularity(args.get("granularity"))
        if granularity is None and action.startswith("default_"):
            granularity = "small"
        if granularity is not None:
            return f"{action}({granularity})"
    return action


def load_actions(response_path: Optional[Path]) -> List[str]:
    if response_path is None or not response_path.exists():
        return []
    data = load_json(response_path)
    actions = []
    for item in data.get("responses", []):
        action = format_action(item)
        if action is not None:
            actions.append(action)
    return actions


def export_results_csv(output_path: Optional[str] = None, merge_existing: Optional[bool] = None) -> Optional[Path]:
    if merge_existing is None:
        merge_existing = not ARGS.no_merge_csv

    extra_fields: List[str] = []
    seen_fields = set()
    for cfg in CSV_EXPORT_CONFIGS:
        for field in cfg["fields"]:
            if field not in seen_fields:
                seen_fields.add(field)
                extra_fields.append(field)

    rows: List[Dict[str, Any]] = []
    for cfg in CSV_EXPORT_CONFIGS:
        for task_dir in task_dirs_for(cfg):
            summary_file, summary = find_export_summary_file(task_dir, cfg)
            if summary_file is None or summary is None:
                continue

            response_file = find_matching_response_file(task_dir, cfg["response_pattern"], summary_file)
            actions = load_actions(response_file)
            actions_text = ";".join(actions)
            row: Dict[str, Any] = {
                "Game": cfg["game"],
                "Task ID": f"{cfg['game']}/{task_dir.name}",
                "Completed": summary.get(cfg["completed_key"], ""),
                "actual_actions": actions_text,
                "actual_actions_count": len(actions),
                "actions": actions_text,
            }
            for field in extra_fields:
                row[field] = ""
            for field in cfg["fields"]:
                row[field] = summary.get(field, "")
            rows.append(row)

    output = Path(output_path) if output_path else BASE_DIR / "results_summary.csv"
    kept_rows: List[Dict[str, Any]] = []
    if merge_existing and output.exists():
        existing_rows = load_existing_csv_rows(output)
        new_task_ids = {row["Task ID"] for row in rows}
        kept_rows = [row for row in existing_rows if row["Task ID"] not in new_task_ids]
        if kept_rows:
            print(f"\nMerged existing CSV rows: kept {len(kept_rows)} unchanged rows.")

    merged_rows = kept_rows + rows
    if not merged_rows:
        print("\nNo exportable evaluation results found; CSV export skipped.")
        return None

    merged_rows.sort(key=csv_sort_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = merge_csv_fieldnames(merged_rows)
    normalized = [normalize_csv_row(row, fieldnames) for row in merged_rows]

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)

    print(f"\nCSV exported: {output}")
    print(f"  new rows: {len(rows)} | kept rows: {len(kept_rows)} | total rows: {len(normalized)}")
    return output


def latest_summaries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries = []
    missing = []
    for task_dir in task_dirs_for(cfg):
        summary_file, summary = find_export_summary_file(task_dir, cfg)
        if summary_file is None or summary is None:
            missing.append(task_dir.name)
        else:
            summaries.append(summary)
    if missing:
        report_missing(cfg["game"].upper(), missing)
    return summaries


def analyze_block3d() -> None:
    cfg = next(c for c in CSV_EXPORT_CONFIGS if c["game"] == "block3d")
    results = latest_summaries(cfg)
    if not results:
        return
    won = sum(1 for row in results if row.get("won", False))
    avg_completion = sum(row.get("completion_percentage", 0) for row in results) / len(results)
    avg_steps = sum(row.get("steps_taken", 0) for row in results) / len(results)
    total_matched = sum(row.get("matched_blocks", 0) for row in results)
    total_target = sum(row.get("target_blocks", 0) for row in results)

    print("=" * 60)
    print("BLOCK3D")
    print("=" * 60)
    print(f"Tasks: {len(results)}")
    print(f"Completed: {fmt_count(won, len(results))}")
    print(f"Average completion: {avg_completion:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Matched blocks: {total_matched}/{total_target} ({100 * total_matched / max(total_target, 1):.1f}%)")


def analyze_maze() -> None:
    cfg = next(c for c in CSV_EXPORT_CONFIGS if c["game"] == "maze")
    results = latest_summaries(cfg)
    if not results:
        return
    success = sum(1 for row in results if row.get("success", False))
    avg_steps = sum(row.get("steps_taken", 0) for row in results) / len(results)

    print("\n" + "=" * 60)
    print("MAZE")
    print("=" * 60)
    print(f"Tasks: {len(results)}")
    print(f"Completed: {fmt_count(success, len(results))}")
    print(f"Average steps: {avg_steps:.1f}")


def analyze_maze3d_pro() -> None:
    cfg = next(c for c in CSV_EXPORT_CONFIGS if c["game"] == "maze3d_pro")
    level_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    missing = []
    for task_dir in task_dirs_for(cfg):
        files = sorted(task_dir.glob(cfg["summary_pattern"]))
        if not files:
            missing.append(task_dir.name)
            continue
        for file in files:
            data = load_json(file)
            level_results[int(data.get("level_number", 0) or 0)].append(data)
    if missing:
        report_missing("MAZE3D_PRO", missing)
    if not level_results:
        return

    best_results = {
        level: sorted(
            runs,
            key=lambda row: (
                not bool(row.get("success")),
                summary_step_count(row),
            ),
        )[0]
        for level, runs in level_results.items()
    }
    success = sum(1 for row in best_results.values() if row.get("success"))
    all_runs = [row for runs in level_results.values() for row in runs]
    success_all = sum(1 for row in all_runs if row.get("success"))

    print("\n" + "=" * 60)
    print("MAZE3D_PRO")
    print("=" * 60)
    print(f"Tasks: {len(best_results)}")
    print(f"Completed, best run per level: {fmt_count(success, len(best_results))}")
    print(f"Completed, all runs: {fmt_count(success_all, len(all_runs))}")
    multi_run = sorted(level for level, runs in level_results.items() if len(runs) > 1)
    if multi_run:
        print(f"Levels with multiple runs: {multi_run}")


def analyze_rubik() -> None:
    cfg = next(c for c in CSV_EXPORT_CONFIGS if c["game"] == "rubik")
    results = latest_summaries(cfg)
    if not results:
        return
    solved = sum(1 for row in results if row.get("cube_solved", False))
    avg_steps = sum(row.get("steps", 0) for row in results) / len(results)

    print("\n" + "=" * 60)
    print("RUBIK")
    print("=" * 60)
    print(f"Tasks: {len(results)}")
    print(f"Completed: {fmt_count(solved, len(results))}")
    print(f"Average steps: {avg_steps:.1f}")


def analyze_snake() -> None:
    cfg = next(c for c in CSV_EXPORT_CONFIGS if c["game"] == "snake")
    results = latest_summaries(cfg)
    if not results:
        return
    scores = [row.get("score", 0) for row in results]
    lengths = [row.get("snake_length", 0) for row in results]
    steps = [row.get("steps_taken", 0) for row in results]

    print("\n" + "=" * 60)
    print("SNAKE")
    print("=" * 60)
    print(f"Runs: {len(results)}")
    print(f"Score: max={max(scores)}, min={min(scores)}, avg={sum(scores) / len(scores):.1f}")
    print(f"Snake length: max={max(lengths)}, min={min(lengths)}, avg={sum(lengths) / len(lengths):.1f}")
    print(f"Steps: max={max(steps)}, min={min(steps)}, avg={sum(steps) / len(steps):.1f}")


def print_summary() -> None:
    rows = []
    for cfg in CSV_EXPORT_CONFIGS:
        summaries = latest_summaries(cfg)
        if not summaries:
            continue
        completed_key = cfg["completed_key"]
        if cfg["game"] == "snake":
            scores = [row.get("score", 0) for row in summaries]
            rows.append((cfg["game"], len(summaries), f"avg score {sum(scores) / len(scores):.1f}", f"max score {max(scores)}"))
        else:
            completed = sum(1 for row in summaries if row.get(completed_key))
            rows.append((cfg["game"], len(summaries), f"{completed}/{len(summaries)}", ""))

    if not rows:
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Game':<14} {'Tasks':>8} {'Result':>14}  Note")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 14}  {'-' * 20}")
    for game, count, result, note in rows:
        print(f"  {game:<14} {count:>8} {result:>14}  {note}")


def main() -> None:
    incomplete = collect_all_missing()
    if incomplete:
        print("Evaluation is incomplete; some task directories are missing summary files:")
        for game, missing in incomplete:
            print(f"\n  [{game}] missing {len(missing)}")
            for name in missing:
                print(f"    - {name}")
    else:
        analyze_block3d()
        analyze_maze()
        analyze_maze3d_pro()
        analyze_rubik()
        analyze_snake()
        print_summary()
    export_results_csv(ARGS.csv_path)


if __name__ == "__main__":
    main()
