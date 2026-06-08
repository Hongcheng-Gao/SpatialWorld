#!/usr/bin/env python3
"""
ProcTHOR dual-agent CSV benchmark runner.

Reuse SpatialWorld single-agent benchmark helpers, but execute `mllm_base_agent/dual_agent/procthor/main.py`.
"""

# """
# python \
# "./dual_agent/run_csv_benchmark.py" \
# --csv "./experiments/csv/procthor/Spatial-Annotation-procthor.csv" \
# --config "./experiments/configs/procthor/dual/config_close_gpt-5.yaml" \
# --agent1 "./experiments/configs/procthor/config_close_kimi-k25.yaml" \
# --agent2 "./experiments/configs/procthor/config_close_kimi-k25.yaml"
# """

"""
python "./dual_agent/run_csv_benchmark.py" \
  --csv "./experiments/csv/procthor/Spatial-Annotation-procthor.csv" \
  --config "./experiments/configs/procthor/dual/config_close_Gemini-2.5-pro.yaml"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.procthor.run_benchmark import (
    benchmark_run_dirname,
    count_csv_status,
    determine_failure_reason,
    prepare_runtime_config,
    read_task_ids_from_csv,
    save_task_log,
    update_csv_task_status,
)

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def find_result_json(task_output_dir: str) -> Optional[Path]:
    """Find dual-agent result json in a task output dir."""
    task_path = Path(task_output_dir)
    if not task_path.exists():
        return None

    log_file = task_path / "log.json"
    if log_file.exists():
        return log_file

    dual_episode_files = sorted(task_path.glob("dual_episode_*.json"))
    if dual_episode_files:
        return dual_episode_files[-1]

    episode_files = sorted(task_path.glob("episode_*.json"))
    if episode_files:
        return episode_files[-1]

    return None


def check_task_success(task_output_dir: str) -> bool:
    """Use dual-agent result json directly as success source."""
    result_json = find_result_json(task_output_dir)
    if not result_json or not result_json.exists():
        return False

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        if result_json.name == "log.json":
            return result_data.get("metadata", {}).get("task_result") == "success"
        return bool(result_data.get("success", False))
    except Exception:
        return False


def find_completed_tasks(output_dir: str) -> set:
    """Find completed tasks from previous dual benchmark dirs."""
    completed = set()
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed

    benchmark_dirs = sorted(
        [d for d in output_path.glob("dual_benchmark_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )
    sequential_dirs = sorted(
        [d for d in output_path.glob("dual_benchmark_sequential_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )
    all_dirs = benchmark_dirs + sequential_dirs

    for benchmark_dir in all_dirs:
        for task_dir in benchmark_dir.iterdir():
            if task_dir.is_dir() and task_dir.name not in {"task_logs", "failed_logs"}:
                if find_result_json(str(task_dir)):
                    completed.add(task_dir.name)
    return completed


def copytree_no_delete(source: Path, dest: Path) -> Path:
    """Copy directory without deleting existing targets."""
    final_dest = dest
    if final_dest.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dest = dest.parent / f"{dest.name}_{suffix}"
    shutil.copytree(source, final_dest)
    return final_dest


def copy_to_outputs_completed(task_id: str, task_output_dir: str, outputs_completed_dir: str) -> bool:
    """Archive completed task outputs without deleting existing data."""
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            return False
        copytree_no_delete(source_path, Path(outputs_completed_dir) / task_id)
        return True
    except Exception as e:
        print(f"  ❌ Copy to outputs_completed failed ({task_id}): {e}")
        return False


def save_failed_snapshot(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int = 1):
    """Save failed task outputs without deleting previous snapshots."""
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            return
        failed_logs_path = Path(failed_logs_dir)
        failed_logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        copytree_no_delete(source_path, failed_logs_path / f"{task_id}_attempt_{attempt}_{timestamp}")
    except Exception as e:
        print(f"  ⚠️  Error saving failed snapshot: {e}")


def write_missing_result_diagnostic(task_output_dir: str, task_id: str, return_code: Optional[int], stdout_text: str, stderr_text: str):
    """Write a diagnostic file when no result json is produced."""
    try:
        diag_path = Path(task_output_dir) / "run_error.txt"
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"Task ID: {task_id}\n")
            if return_code is not None:
                f.write(f"Return code: {return_code}\n")
            f.write(f"{'=' * 80}\n")
            f.write("No result JSON found in task output directory.\n\n")
            if stdout_text:
                f.write("=== STDOUT ===\n")
                f.write(stdout_text[-10000:])
                f.write("\n\n")
            if stderr_text:
                f.write("=== STDERR ===\n")
                f.write(stderr_text[-10000:])
        print(f"  📝 Diagnostic written: {diag_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to write diagnostic: {e}")


def main():
    parser = argparse.ArgumentParser(description="ProcTHOR dual-agent CSV benchmark runner")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with task IDs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--config", type=str, default="experiments/configs/procthor/dual/config_close_gpt-5.yaml", help="Base config path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output root dir")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--task", type=str, default=None, help="Run only one task")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument("--skip-completed", action="store_true", help="Skip already completed tasks")
    parser.add_argument("--outputs-completed-dir", type=str, default="outputs_completed", help="Archive dir")
    parser.add_argument("--save-name", type=str, default=None, help="Benchmark run dir suffix")
    parser.add_argument("--switch-interval", type=int, default=1, help="Agent switch interval")
    parser.add_argument("--collaboration-mode", type=str, default="alternating", choices=["alternating", "sequential"], help="Collaboration mode")
    parser.add_argument("--agent1", type=str, default=None, help="Agent 1 single-agent config path")
    parser.add_argument("--agent2", type=str, default=None, help="Agent 2 single-agent config path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    if args.agent1 and not Path(args.agent1).exists():
        print(f"❌ Agent 1 config not found: {args.agent1}")
        sys.exit(1)
    if args.agent2 and not Path(args.agent2).exists():
        print(f"❌ Agent 2 config not found: {args.agent2}")
        sys.exit(1)

    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    else:
        print(f"📋 Reading task IDs from CSV: {csv_path}")
        task_ids = read_task_ids_from_csv(str(csv_path), only_null=True)
        if not task_ids:
            print("❌ No task IDs with Completed=null in CSV")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks with Completed=null")

    if args.skip_completed:
        completed_tasks = find_completed_tasks(args.output_dir)
        if completed_tasks:
            original_count = len(task_ids)
            task_ids = [tid for tid in task_ids if tid not in completed_tasks]
            print(f"✓ Skipping {original_count - len(task_ids)} completed tasks")
            if not task_ids:
                print("✓ All tasks completed, nothing to run")
                sys.exit(0)

    actual_config_path, temp_config_file, headless_mode = prepare_runtime_config(
        config_path=config_path,
        headless=args.headless,
    )
    if args.headless:
        print(f"🖥️  Headless mode: {headless_mode}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "dual_benchmark_sequential" if args.sequential else "dual_benchmark"
    benchmark_output_dir = Path(args.output_dir) / benchmark_run_dirname(prefix, timestamp, args.save_name)
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    task_logs_dir = benchmark_output_dir / "task_logs"
    task_logs_dir.mkdir(parents=True, exist_ok=True)
    failed_logs_dir = benchmark_output_dir / "failed_logs"
    failed_logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_completed_path = Path(args.outputs_completed_dir)
    outputs_completed_path.mkdir(parents=True, exist_ok=True)
    csv_lock = Lock()

    print(f"\n{'=' * 80}")
    print(f"🚀 Starting {'sequential' if args.sequential else 'parallel'} ProcTHOR dual-agent benchmark")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Workers: {1 if args.sequential else args.workers}")
    print(f"Config: {actual_config_path}")
    if args.agent1:
        print(f"Agent 1 config: {args.agent1}")
    if args.agent2:
        print(f"Agent 2 config: {args.agent2}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"{'=' * 80}\n")

    def execute_task(task_id: str) -> dict:
        task_start_time = time.time()
        task_log_parts = []
        task_status = "failed_external"
        task_success = False

        task_output_dir = benchmark_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "mllm_base_agent.dual_agent.procthor.main",
            "--config",
            str(actual_config_path),
            "--tasks",
            task_id,
            "--output-dir",
            str(task_output_dir),
            "--switch-interval",
            str(args.switch_interval),
            "--collaboration-mode",
            args.collaboration_mode,
        ]
        if args.agent1:
            cmd.extend(["--agent1", args.agent1])
        if args.agent2:
            cmd.extend(["--agent2", args.agent2])
        if args.headless:
            cmd.append("--headless")
        if args.max_steps:
            cmd.extend(["--max-steps", str(args.max_steps)])

        try:
            execution_start_time = time.time()
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=None,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            execution_duration = time.time() - execution_start_time

            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""
            if stdout_text:
                task_log_parts.append("=== STDOUT ===\n")
                task_log_parts.append(stdout_text)
                task_log_parts.append("\n")
            if stderr_text:
                task_log_parts.append("=== STDERR ===\n")
                task_log_parts.append(stderr_text)
                task_log_parts.append("\n")
            task_log_parts.append("=== Run info ===\n")
            task_log_parts.append(f"Exit code: {result.returncode}\n")
            task_log_parts.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_parts.append(f"{'=' * 80}\n\n")
            task_log = "".join(task_log_parts)

            result_json = find_result_json(str(task_output_dir))
            if check_task_success(str(task_output_dir)):
                task_success = True
                task_status = "success"
                save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                if copy_to_outputs_completed(task_id, str(task_output_dir), str(outputs_completed_path)):
                    update_csv_task_status(csv_path, task_id, "true", csv_lock)
                print(f"  ✅ {task_id} success")
            else:
                failure_reason = determine_failure_reason(result_json_path=result_json)
                save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                save_failed_snapshot(task_id, str(task_output_dir), str(failed_logs_dir), 1)
                if result_json:
                    if failure_reason in ("api_error", "env_error", "external_error"):
                        task_status = "failed_external"
                        print(f"  ⚠️  {task_id} external failure -> null")
                    else:
                        task_status = "failed_model"
                        if copy_to_outputs_completed(task_id, str(task_output_dir), str(outputs_completed_path)):
                            update_csv_task_status(csv_path, task_id, "false", csv_lock)
                        print(f"  ❌ {task_id} model failure -> false")
                else:
                    task_status = "failed_external"
                    write_missing_result_diagnostic(str(task_output_dir), task_id, result.returncode, stdout_text, stderr_text)
                    print(f"  ⚠️  {task_id} no result json -> null")
        except Exception as e:
            task_status = "failed_external"
            task_log_parts.append(f"=== Exception ===\nError: {e}\n")
            task_log = "".join(task_log_parts)
            save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
            save_failed_snapshot(task_id, str(task_output_dir), str(failed_logs_dir), 1)
            print(f"  ❌ {task_id} exception: {e}")

        return {
            "task_id": task_id,
            "status": task_status,
            "duration": time.time() - task_start_time,
            "success": task_success,
        }

    task_records = []
    successful = 0
    failed_model = 0
    failed_external = 0

    try:
        if args.sequential:
            task_iterator = tqdm(task_ids, desc="Tasks", unit="task", ncols=100) if HAS_TQDM else task_ids
            for task_id in task_iterator:
                result = execute_task(task_id)
                task_records.append(result)
                if result["success"]:
                    successful += 1
                elif result["status"] == "failed_model":
                    failed_model += 1
                else:
                    failed_external += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(execute_task, task_id): task_id for task_id in task_ids}
                iterator = tqdm(as_completed(future_to_task), total=len(task_ids), desc="Tasks", unit="task", ncols=100) if HAS_TQDM else as_completed(future_to_task)
                for future in iterator:
                    result = future.result()
                    task_records.append(result)
                    if result["success"]:
                        successful += 1
                    elif result["status"] == "failed_model":
                        failed_model += 1
                    else:
                        failed_external += 1
                    if HAS_TQDM:
                        iterator.set_postfix({"ok": successful, "model": failed_model, "external": failed_external})
    finally:
        if temp_config_file and os.path.exists(temp_config_file.name):
            try:
                os.unlink(temp_config_file.name)
            except Exception:
                pass

    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    total_duration = sum(record["duration"] for record in task_records)
    avg_duration = total_duration / len(task_records) if task_records else 0
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write("Dual-agent benchmark summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Config: {config_path}\n")
        if args.agent1:
            f.write(f"Agent 1 config: {args.agent1}\n")
        if args.agent2:
            f.write(f"Agent 2 config: {args.agent2}\n")
        f.write(f"Output dir: {benchmark_output_dir}\n")
        f.write(f"Total tasks: {len(task_ids)}\n")
        f.write(f"Success: {successful}\n")
        f.write(f"Model failure: {failed_model}\n")
        f.write(f"External failure: {failed_external}\n")
        f.write(f"Total time: {total_duration:.2f}s\n")
        f.write(f"Avg time: {avg_duration:.2f}s\n")

    csv_stats = count_csv_status(csv_path)
    print(f"\n{'=' * 80}")
    print("🎉 ProcTHOR dual-agent benchmark complete")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Success: {successful}")
    print(f"Model failure: {failed_model}")
    print(f"External failure: {failed_external}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Summary log: {summary_log_path}")
    print(f"CSV status: true={csv_stats['true']} false={csv_stats['false']} null={csv_stats['null']}")
    print(f"{'=' * 80}\n")
    sys.exit(0 if (failed_model + failed_external) == 0 else 1)


if __name__ == "__main__":
    main()
