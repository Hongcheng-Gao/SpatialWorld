#!/usr/bin/env python3
"""Run all game benchmarks and export a combined CSV summary."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys
import time
from typing import Any, Dict, List, Optional


GAMES = [
    {
        "name": "maze",
        "script": "maze_openai_evaluation.py",
        "mode": "level",
        "levels": list(range(1, 21)),
    },
    {
        "name": "block3d",
        "script": "block3d_openai_evaluation.py",
        "mode": "level",
        "levels": list(range(1, 21)),
    },
    {
        "name": "maze3d_pro",
        "script": "maze3d_pro_openai_evaluation.py",
        "mode": "level",
        "levels": list(range(1, 26)),
    },
    {
        "name": "rubik",
        "script": "rubik_openai_evaluation.py",
        "mode": "level",
        "levels": list(range(1, 21)),
    },
    {
        "name": "snake",
        "script": "snake_openai_evaluation.py",
        "mode": "run",
        "runs": 20,
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLLM game benchmarks and export results_summary.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python -m scripts.game.run_benchmark --output-dir outputs/game_exp1
  python -m scripts.game.run_benchmark --output-dir outputs/game_exp1 --model gpt-4o --parallelism 6
  python -m scripts.game.run_benchmark --output-dir outputs/game_exp1 --games maze,rubik
  python -m scripts.game.run_benchmark --output-dir outputs/game_exp1 \\
      --model gpt-4o --api-base-url https://example.invalid/v1 --api-key $env:OPENAI_API_KEY

Available games: {", ".join(g["name"] for g in GAMES)}
""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: logs/allgames_<timestamp>.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Maximum concurrent game tasks. Default: 4.",
    )
    parser.add_argument(
        "--games",
        type=str,
        default=None,
        help="Comma-separated game names to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override per-game max steps.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Override OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key. Prefer OPENAI_API_KEY in the environment.",
    )
    parser.add_argument(
        "--retry-times",
        type=int,
        default=None,
        help="Override API retry count.",
    )
    return parser.parse_args()


def append_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.model is not None:
        cmd += ["--model", args.model]
    if args.api_base_url is not None:
        cmd += ["--api-base-url", args.api_base_url]
    if args.api_key is not None:
        cmd += ["--api-key", args.api_key]
    if args.retry_times is not None:
        cmd += ["--retry-times", str(args.retry_times)]


def build_cmd_level(script_path: str, level: int, log_dir: str, args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, script_path, "--level", str(level), "--log-dir", log_dir]
    append_common_args(cmd, args)
    return cmd


def build_cmd_run(script_path: str, log_dir: str, args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, script_path, "--log-dir", log_dir]
    append_common_args(cmd, args)
    return cmd


def is_task_completed(log_dir: str) -> bool:
    """A task is complete when it has written an evaluation summary JSON."""
    if not os.path.exists(log_dir):
        return False
    return any(
        "_evaluation_summary_" in name and name.endswith(".json")
        for name in os.listdir(log_dir)
    )


async def run_task(
    sem: asyncio.Semaphore,
    cmd: List[str],
    label: str,
    log_dir: str,
) -> Dict[str, Any]:
    os.makedirs(log_dir, exist_ok=True)

    async with sem:
        start_time = time.time()
        print(f"  [{label}] start")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            duration = time.time() - start_time
            status = "OK" if proc.returncode == 0 else "FAIL"
            print(f"  [{label}] {status} | duration: {duration:.1f}s | return code: {proc.returncode}")
            return {
                "label": label,
                "returncode": proc.returncode,
                "duration": duration,
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            }
        except Exception as exc:
            duration = time.time() - start_time
            print(f"  [{label}] FAIL: {exc}")
            return {
                "label": label,
                "returncode": -1,
                "duration": duration,
                "stdout": "",
                "stderr": str(exc),
            }


async def run_analyze_and_export(root_output_dir: str) -> int:
    analyze_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analyze_and_export.py",
    )
    if not os.path.exists(analyze_script):
        print(f"analyze_and_export.py not found: {analyze_script}")
        return 1

    print()
    print("Running analyze_and_export.py ...")
    print(f"  {sys.executable} {analyze_script} {root_output_dir}")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        analyze_script,
        root_output_dir,
    )
    returncode = await proc.wait()
    if returncode == 0:
        print(f"CSV exported: {os.path.join(root_output_dir, 'results_summary.csv')}")
    else:
        print(f"analyze_and_export.py failed with return code {returncode}")
    return returncode


def select_games(games_arg: Optional[str]) -> List[Dict[str, Any]]:
    if not games_arg:
        return GAMES

    selected_names = {name.strip() for name in games_arg.split(",") if name.strip()}
    valid_names = {game["name"] for game in GAMES}
    invalid = selected_names - valid_names
    if invalid:
        raise SystemExit(
            f"Invalid games: {', '.join(sorted(invalid))}. "
            f"Available games: {', '.join(sorted(valid_names))}"
        )
    return [game for game in GAMES if game["name"] in selected_names]


def build_tasks(games_to_run: List[Dict[str, Any]], scripts_dir: str, root_output_dir: str, args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[str]]:
    all_tasks: List[Dict[str, Any]] = []
    skipped_games: List[str] = []

    for game in games_to_run:
        script_path = os.path.join(scripts_dir, game["script"])
        if not os.path.exists(script_path):
            print(f"[{game['name']}] missing script: {script_path}; skipped")
            skipped_games.append(game["name"])
            continue

        game_output_dir = os.path.join(root_output_dir, game["name"])
        os.makedirs(game_output_dir, exist_ok=True)

        if game["mode"] == "level":
            for level in game["levels"]:
                log_dir = os.path.join(game_output_dir, f"level_{level:02d}")
                all_tasks.append(
                    {
                        "game": game["name"],
                        "label": f"{game['name']}/level_{level:02d}",
                        "log_dir": log_dir,
                        "cmd": build_cmd_level(script_path, level, log_dir, args),
                    }
                )
        else:
            for run_idx in range(game["runs"]):
                log_dir = os.path.join(game_output_dir, f"run_{run_idx + 1:03d}")
                all_tasks.append(
                    {
                        "game": game["name"],
                        "label": f"{game['name']}/run_{run_idx + 1:03d}",
                        "log_dir": log_dir,
                        "cmd": build_cmd_run(script_path, log_dir, args),
                    }
                )

    return all_tasks, skipped_games


def print_plan(games_to_run: List[Dict[str, Any]], args: argparse.Namespace, root_output_dir: str) -> None:
    print("=" * 60)
    print("MLLM game benchmark")
    print("=" * 60)
    print(f"Games: {len(games_to_run)}")
    print(f"Parallelism: {args.parallelism}")
    print(f"Output dir: {root_output_dir}")
    if args.model:
        print(f"Model: {args.model}")
    if args.max_steps:
        print(f"Max steps: {args.max_steps}")
    print()
    for game in games_to_run:
        if game["mode"] == "level":
            print(f"  {game['name']:<14} levels {game['levels'][0]}-{game['levels'][-1]} ({len(game['levels'])})")
        else:
            print(f"  {game['name']:<14} runs {game['runs']}")
    print("=" * 60)


def count_previously_done(game: Dict[str, Any], root_output_dir: str, pending_tasks: List[Dict[str, Any]]) -> int:
    pending_labels = {task["label"] for task in pending_tasks}
    game_dir = os.path.join(root_output_dir, game["name"])
    if game["mode"] == "level":
        return sum(
            1
            for level in game["levels"]
            if is_task_completed(os.path.join(game_dir, f"level_{level:02d}"))
            and f"{game['name']}/level_{level:02d}" not in pending_labels
        )
    return sum(
        1
        for run_idx in range(game["runs"])
        if is_task_completed(os.path.join(game_dir, f"run_{run_idx + 1:03d}"))
        and f"{game['name']}/run_{run_idx + 1:03d}" not in pending_labels
    )


async def main() -> None:
    args = parse_arguments()
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    root_output_dir = args.output_dir or os.path.join(
        "logs",
        f"allgames_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(root_output_dir, exist_ok=True)

    games_to_run = select_games(args.games)
    print_plan(games_to_run, args, root_output_dir)

    all_tasks, skipped_games = build_tasks(games_to_run, scripts_dir, root_output_dir, args)
    pending_tasks = [task for task in all_tasks if not is_task_completed(task["log_dir"])]
    skipped_count = len(all_tasks) - len(pending_tasks)

    if skipped_count:
        print(f"\nResume: skipped {skipped_count} completed tasks; pending {len(pending_tasks)}")
        for task in all_tasks:
            if is_task_completed(task["log_dir"]):
                print(f"  [done] {task['label']}")
    else:
        print(f"\nScheduled {len(all_tasks)} tasks with parallelism {args.parallelism}.")
    print("=" * 60)

    if not pending_tasks:
        print("All tasks already completed.")
        await run_analyze_and_export(root_output_dir)
        return

    total_start = time.time()
    sem = asyncio.Semaphore(args.parallelism)
    results = await asyncio.gather(
        *(run_task(sem, task["cmd"], task["label"], task["log_dir"]) for task in pending_tasks)
    )

    game_results: Dict[str, List[Dict[str, Any]]] = {}
    for task, result in zip(pending_tasks, results):
        game_results.setdefault(task["game"], []).append(result)

    print(f"\n{'=' * 60}")
    print("Benchmark summary")
    print("=" * 60)
    for game in games_to_run:
        if game["name"] in skipped_games:
            print(f"{game['name']:<14} skipped; script missing")
            continue

        current_results = game_results.get(game["name"], [])
        ok = sum(1 for result in current_results if result["returncode"] == 0)
        fail = len(current_results) - ok
        pre_done = count_previously_done(game, root_output_dir, pending_tasks)
        print(f"{game['name']:<14} OK {ok + pre_done} | failed {fail}")
        failed_labels = [result["label"] for result in current_results if result["returncode"] != 0]
        if failed_labels:
            print(f"  Failed tasks: {', '.join(failed_labels)}")

    print(f"Duration: {time.time() - total_start:.1f}s")
    print(f"Output dir: {root_output_dir}")
    await run_analyze_and_export(root_output_dir)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
