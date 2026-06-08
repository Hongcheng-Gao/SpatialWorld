#!/usr/bin/env python3
"""
Read task IDs from CSV and run ProcTHOR benchmark (parallel or sequential).

  spatial-planning       ：
-    python -m scripts.procthor.work.run_task      （VLM   ，  log.json）
- check_task_success：VLM     log     metadata.task_result；golden      evaluate_action_sequence
-    procthor    ID
"""

import os
import sys
import csv
import json
import subprocess
import yaml
import tempfile
import shutil
import time
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed, using simple progress. Install: pip install tqdm")


#   AI2-THOR benchmark   ：            ，   action_eval_*     
EVAL_LOCK = Lock()


def sanitize_benchmark_save_name(name: str) -> str:
    """        ，              """
    s = name.strip()
    for bad in ("/", "\\", ":", "\x00"):
        s = s.replace(bad, "_")
    return s[:200] if s else ""


def benchmark_run_dirname(prefix: str, timestamp: str, save_name: Optional[str]) -> str:
    """
    prefix: 'benchmark' | 'benchmark_sequential'
      save_name  ：{prefix}_{save_name}_{   }（     ）
      save_name  ：{prefix}_{   }
    """
    safe = sanitize_benchmark_save_name(save_name) if save_name else ""
    if safe:
        return f"{prefix}_{safe}_{timestamp}"
    return f"{prefix}_{timestamp}"


def prepare_runtime_config(config_path: Path, headless: bool) -> tuple[Path, Optional[tempfile.NamedTemporaryFile], str]:
    """
    Prepare runtime config for benchmark execution.

    Behavior:
    - headless=False: use original config directly.
    - headless=True + DISPLAY available: prefer X display path (xvfb/remote display),
      avoid forcing CloudRendering (which may require Vulkan/libvulkan1).
    - headless=True + no DISPLAY: fallback to CloudRendering.

    Returns:
        (actual_config_path, temp_config_file, headless_mode)
        headless_mode in {"disabled", "x_display", "cloudrendering"}.
    """
    if not headless:
        return config_path, None, "disabled"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if "env" not in config_data or not isinstance(config_data["env"], dict):
        config_data["env"] = {}

    env_cfg = config_data["env"]
    display = os.environ.get("DISPLAY", "").strip()
    x_display_cfg = str(env_cfg.get("x_display", "")).strip()
    runtime_x_display = x_display_cfg or display

    if runtime_x_display:
        # Prefer X display under xvfb instead of CloudRendering to avoid Vulkan dependency.
        env_cfg["x_display"] = runtime_x_display
        platform_cfg = env_cfg.get("platform")
        if isinstance(platform_cfg, str) and platform_cfg.lower() == "cloudrendering":
            env_cfg["platform"] = None
        headless_mode = "x_display"
    else:
        env_cfg["platform"] = "CloudRendering"
        headless_mode = "cloudrendering"

    temp_config_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
    temp_config_file.close()

    return Path(temp_config_file.name), temp_config_file, headless_mode


def read_task_ids_from_csv(csv_path: str, only_null: bool = True) -> list:
    """
    Read task IDs from CSV.

    Args:
        csv_path: Path to CSV.
        only_null: If True, only include rows with Completed null/empty.

    Returns:
        List of task IDs.
    """
    task_ids = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        completed_col_idx = None
        if header and len(header) > 1:
            for idx, col in enumerate(header):
                if col.strip().lower() in ['completed', 'status']:
                    completed_col_idx = idx
                    break
        
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            task_id = row[0].strip()
            if task_id == "Task ID":
                continue

            if only_null:
                if completed_col_idx is not None and len(row) > completed_col_idx:
                    status = row[completed_col_idx].strip().lower()
                    if status in ["true", "false"]:
                        continue
                task_ids.append(task_id)
            else:
                task_ids.append(task_id)
    
    return task_ids


def find_completed_tasks(output_dir: str) -> set:
    """
    Find completed tasks (log.json or episode_*.json in latest benchmark dirs).

    Args:
        output_dir: Output root dir.

    Returns:
        Set of completed task IDs.
    """
    completed = set()
    output_path = Path(output_dir)

    if not output_path.exists():
        return completed

    benchmark_dirs = sorted(
        [d for d in output_path.glob("benchmark_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    sequential_dirs = sorted(
        [d for d in output_path.glob("benchmark_sequential_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    all_dirs = benchmark_dirs + sequential_dirs

    for benchmark_dir in all_dirs:
        for worker_dir in benchmark_dir.glob("worker_*"):
            if worker_dir.is_dir():
                for task_dir in worker_dir.iterdir():
                    if task_dir.is_dir():
                        task_id = task_dir.name
                        has_log = (task_dir / "log.json").exists()
                        has_episode = bool(list(task_dir.glob("episode_*.json")))
                        
                        if has_log or has_episode:
                            completed.add(task_id)
                        else:
                            for run_dir in task_dir.glob("run_*"):
                                if run_dir.is_dir():
                                    task_subdir = run_dir / task_id
                                    if task_subdir.exists():
                                        if (task_subdir / "log.json").exists() or list(task_subdir.glob("episode_*.json")):
                                            completed.add(task_id)
                                            break
                                    if (run_dir / "log.json").exists() or list(run_dir.glob("episode_*.json")):
                                        completed.add(task_id)
                                        break

        for task_dir in benchmark_dir.iterdir():
            if task_dir.is_dir() and not task_dir.name.startswith("worker_"):
                task_id = task_dir.name
                log_file = task_dir / "log.json"
                episode_files = list(task_dir.glob("episode_*.json"))
                
                for run_dir in task_dir.glob("run_*"):
                    if run_dir.is_dir():
                        task_subdir = run_dir / task_id
                        if task_subdir.exists():
                            if (task_subdir / "log.json").exists() or list(task_subdir.glob("episode_*.json")):
                                completed.add(task_id)
                                break
                        if (run_dir / "log.json").exists() or list(run_dir.glob("episode_*.json")):
                            completed.add(task_id)
                            break
                
                if log_file.exists() or episode_files:
                    completed.add(task_id)
    
    return completed


def find_result_json(task_output_dir: str) -> Optional[Path]:
    """
    Find result JSON (log.json or episode_*.json) in task output dir.

    Returns:
        Path to JSON or None.
    """
    task_path = Path(task_output_dir)
    log_file = task_path / "log.json"
    if log_file.exists():
        return log_file
    
    episode_files = list(task_path.glob("episode_*.json"))
    if episode_files:
        return episode_files[0]
    
    for run_dir in task_path.glob("run_*"):
        if run_dir.is_dir():
            # Try task_path.name (may have retry suffix)
            task_subdir_with_retry = run_dir / task_path.name
            if task_subdir_with_retry.exists():
                log_file = task_subdir_with_retry / "log.json"
                if log_file.exists():
                    return log_file
                episode_files = list(task_subdir_with_retry.glob("episode_*.json"))
                if episode_files:
                    return episode_files[0]
            
            task_name_with_retry = task_path.name
            task_id_match = re.match(r"^(ai2thor\d+|carla\d+|procthor\d+)(?:_retry_\d+)?$", task_name_with_retry)
            if task_id_match:
                task_id = task_id_match.group(1)
                task_subdir = run_dir / task_id
                if task_subdir.exists():
                    log_file = task_subdir / "log.json"
                    if log_file.exists():
                        return log_file
                    episode_files = list(task_subdir.glob("episode_*.json"))
                    if episode_files:
                        return episode_files[0]
            
            for subdir in run_dir.iterdir():
                if subdir.is_dir():
                    log_file = subdir / "log.json"
                    if log_file.exists():
                        return log_file
                    episode_files = list(subdir.glob("episode_*.json"))
                    if episode_files:
                        return episode_files[0]
            
            log_file = run_dir / "log.json"
            if log_file.exists():
                return log_file
            episode_files = list(run_dir.glob("episode_*.json"))
            if episode_files:
                return episode_files[0]
    
    return None


def evaluate_task(task_id: str, config_path: str, headless: bool = False) -> bool:
    """
    Run scripts/evaluate_action_sequence.py to evaluate task success.

    Returns:
        True if evaluation succeeds.
    """
    try:
        with EVAL_LOCK:
            cmd = [
                sys.executable,
                "scripts/evaluate_action_sequence.py",
                "--task", task_id,
            ]
            if headless:
                cmd.append("--headless")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode != 0:
                print(f"  ⚠️  Evaluate command failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:500]}")
                return False
            
            output_dirs = sorted(Path("outputs").glob("action_eval_*"))
            if not output_dirs:
                print(f"  ⚠️  No evaluate output dir found")
                return False
            
            latest_output_dir = output_dirs[-1]
            json_files = sorted(latest_output_dir.glob("action_sequence_result_*.json"))
            
            if not json_files:
                print(f"  ⚠️  No evaluate result JSON found")
                return False
            
            latest_json_file = json_files[-1]
            
            with open(latest_json_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            success = result_data.get("success", False)
            evaluation_score = result_data.get("evaluation_score", 0.0)
            
            if success and evaluation_score == 1.0:
                print(f"  ✅ Evaluate success (score: {evaluation_score})")
                return True
            else:
                print(f"  ❌ Evaluate failed (score: {evaluation_score}, success: {success})")
                return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Evaluate timeout (>5 min)")
        return False
    except Exception as e:
        print(f"  ⚠️  Evaluate error: {e}")
        return False


def check_task_success(task_id: str, task_output_dir: str, config_path: str, headless: bool = False) -> bool:
    """
    Check if task completed successfully (JSON present, then evaluate).
      AI2-THOR benchmark   ：       JSON    ，    evaluate_task          
    """
    result_json = find_result_json(task_output_dir)
    if not result_json:
        return False

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        if result_json.name.startswith('episode_'):
            if not result_data.get('success', False):
                print(f"  ⚠️  Task run failed: {result_data.get('fail_reason', 'Unknown reason')}")
                return False

        if result_json.name == 'log.json':
            metadata = result_data.get('metadata', {})
            task_result = metadata.get('task_result', '')
            if task_result == 'failure':
                print(f"  ⚠️  Task run failed: {metadata.get('fail_reason', 'Unknown reason')}")
                return False
    except Exception as e:
        print(f"  ⚠️  Failed to read task result JSON: {e}")
        return False

    return evaluate_task(task_id, config_path, headless)


def copy_to_outputs_completed(task_id: str, task_output_dir: str, outputs_completed_dir: str) -> bool:
    """
    Copy completed task (true or false) to outputs_completed dir.

    Returns:
        True if copy succeeded.
    """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            print(f"  ⚠️  Source dir not found: {task_output_dir}")
            return False

        dest_path = Path(outputs_completed_dir) / task_id
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
        return True
    except Exception as e:
        print(f"  ❌ Copy to outputs_completed failed ({task_id}): {e}")
        return False


def normalize_task_id(task_id: str) -> str:
    """Normalize task ID (e.g. ai2thor_04000 -> ai2thor04000, procthor_00001 -> procthor00001)."""
    if '_' in task_id and task_id.startswith('ai2thor_'):
        return task_id.replace('ai2thor_', 'ai2thor', 1)
    if '_' in task_id and task_id.startswith('procthor_'):
        return task_id.replace('procthor_', 'procthor', 1)
    return task_id


def determine_failure_reason(task_log_content: str = None, result_json_path: Path = None) -> str:
    """
    Determine failure reason; prefer failure_type from JSON.

    Supports both log.json (metadata.failure_type) and episode_*.json (root failure_type).
    Returns one of: api_error, parse_error, env_error, action_error, model_error, external_error.
    """
    if result_json_path and result_json_path.exists():
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            failure_type = result_data.get('failure_type')
            if failure_type:
                return failure_type
            metadata = result_data.get('metadata', {})
            failure_type = metadata.get('failure_type')
            if failure_type:
                return failure_type
        except Exception as e:
            print(f"  ⚠️  Failed to read failure_type from JSON: {e}")
    return "external_error"


def update_csv_task_status(csv_path: Path, task_id: str, status: str = 'true', lock: Lock = None):
    """Update Completed status for the given task in CSV. Thread-safe if lock provided."""
    if lock:
        lock.acquire()

    try:
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        
        if not rows:
            return False
        
        header = rows[0]
        completed_col_idx = None
        for idx, col in enumerate(header):
            if col.strip().lower() in ['completed', 'status']:
                completed_col_idx = idx
                break

        if completed_col_idx is None:
            header.append("Completed")
            completed_col_idx = len(header) - 1
            for i in range(1, len(rows)):
                if len(rows[i]) <= completed_col_idx:
                    rows[i].extend([''] * (completed_col_idx + 1 - len(rows[i])))
        
        normalized_task_id = normalize_task_id(task_id)
        updated = False
        for i in range(1, len(rows)):
            if not rows[i] or not rows[i][0].strip():
                continue
            csv_task_id = rows[i][0].strip()
            normalized_csv_id = normalize_task_id(csv_task_id)
            if normalized_csv_id == normalized_task_id or csv_task_id == task_id:
                while len(rows[i]) <= completed_col_idx:
                    rows[i].append('')
                old_status = rows[i][completed_col_idx].strip().lower() if rows[i][completed_col_idx] else ''
                if status.lower() == 'null':
                    new_status = ''
                else:
                    new_status = status
                if old_status != status.lower() and (old_status != '' or status.lower() != 'null'):
                    rows[i][completed_col_idx] = new_status
                    updated = True
                    break
        
        if updated:
            backup_path = csv_path.with_suffix('.csv.backup')
            if not backup_path.exists():
                shutil.copy2(csv_path, backup_path)
            
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            return True
        
        return False
    except Exception as e:
        print(f"  ⚠️  Update CSV failed ({task_id}): {e}")
        return False
    finally:
        if lock:
            lock.release()


def save_task_log(task_id: str, log_content: str, logs_dir: Path, attempt: int = None, status: str = "unknown"):
    """Save a single task log file."""
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        if attempt:
            log_filename = f"{task_id}_attempt_{attempt}_{status}.log"
        else:
            log_filename = f"{task_id}_{status}.log"
        log_path = logs_dir / log_filename
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Status: {status}\n")
            if attempt:
                f.write(f"Attempt: {attempt}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(log_content)
        
        return log_path
    except Exception as e:
        print(f"  ⚠️  Save task log failed ({task_id}): {e}")
        return None


def save_failed_log(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int, output_lines: list = None):
    """Save failed task log (copy task output to failed_logs_dir)."""
    try:
        failed_logs_path = Path(failed_logs_dir)
        failed_logs_path.mkdir(parents=True, exist_ok=True)
        task_failed_dir = failed_logs_path / f"{task_id}_attempt_{attempt}"
        if task_failed_dir.exists():
            shutil.rmtree(task_failed_dir)
        
        source_path = Path(task_output_dir)
        if source_path.exists():
            shutil.copytree(source_path, task_failed_dir)
            print(f"  📝 Failed log saved to: {task_failed_dir}")
    except Exception as e:
        print(f"  ⚠️  Error saving failed log: {e}")


def count_csv_status(csv_path: Path) -> dict:
    """Count true/false/null in CSV Completed column. Returns dict with true, false, null, total."""
    stats = {"true": 0, "false": 0, "null": 0, "total": 0}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if not header:
                return stats
            
            completed_col_idx = None
            for idx, col in enumerate(header):
                if col.strip().lower() in ['completed', 'status']:
                    completed_col_idx = idx
                    break
            
            if completed_col_idx is None:
                return stats
            
            for row in reader:
                if not row or not row[0].strip():
                    continue
                
                task_id = row[0].strip()
                if task_id == "Task ID":
                    continue
                
                stats["total"] += 1
                
                if len(row) > completed_col_idx:
                    status = row[completed_col_idx].strip().lower()
                    if status == "true":
                        stats["true"] += 1
                    elif status == "false":
                        stats["false"] += 1
                    else:
                        stats["null"] += 1
                else:
                    stats["null"] += 1
    except Exception as e:
        print(f"  ⚠️  Count CSV status failed: {e}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Read task IDs from CSV and run benchmark (parallel or sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --workers 4 --config experiments/configs/procthor/config_close_gpt-5.yaml
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --workers 4 --config experiments/configs/procthor/config_close_gpt-5.yaml --headless
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --sequential --config experiments/configs/procthor/config_close_gpt-5.yaml --headless
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --sequential --config experiments/configs/procthor/config_close_gpt-5.yaml --skip-completed
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --workers 1 --config experiments/configs/procthor/config_close_gpt-5.yaml --task procthor00001
  python -m scripts.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor.csv" --config experiments/configs/procthor/config_close_gpt-5.yaml --save-name "MyModel"
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with task IDs"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/procthor/config_open_GLM-4.6V.yaml",
        help="Config file path (default: experiments/configs/procthor/config_open_GLM-4.6V.yaml)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output root dir (default: outputs)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode (auto: use X display when DISPLAY exists, otherwise CloudRendering)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tasks sequentially (one by one). Use when API has concurrency limits"
    )
    
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip already completed tasks (from latest benchmark dir)"
    )
    
    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default="outputs_completed",
        help="Output dir for completed tasks (default: outputs_completed)"
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional tag in benchmark output dir name (under --output-dir), e.g. outputs/benchmark_<save-name>_<timestamp>",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only this task ID (e.g. procthor000). Otherwise run all null tasks in CSV"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    else:
        print(f"📋 Reading task IDs from CSV: {csv_path}")
        task_ids = read_task_ids_from_csv(str(csv_path), only_null=True)
        if not task_ids:
            print("❌ No task IDs with Completed=null in CSV")
            print("💡 All tasks may be true/false, or CSV format may be wrong")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks with Completed=null")
    print(f"  First 5: {task_ids[:5]}")
    if len(task_ids) > 5:
        print(f"  ... (total {len(task_ids)} tasks)")
    
    if args.skip_completed:
        print(f"\n🔍 Checking completed tasks (from latest benchmark dir)...")
        completed_tasks = find_completed_tasks(args.output_dir)
        
        if completed_tasks:
            print(f"✓ Found {len(completed_tasks)} completed tasks (with JSON)")
            original_count = len(task_ids)
            task_ids = [tid for tid in task_ids if tid not in completed_tasks]
            skipped_count = original_count - len(task_ids)
            
            if skipped_count > 0:
                print(f"✓ Skipping {skipped_count} completed tasks")
                print(f"✓ Remaining: {len(task_ids)} tasks to run")
                if len(completed_tasks) <= 10:
                    print(f"  Completed: {sorted(completed_tasks)}")
                else:
                    completed_list = sorted(list(completed_tasks))
                    print(f"  Completed (first 10): {completed_list[:10]}...")
                    print(f"  (total {len(completed_tasks)} completed)")
                print(f"\n💡 Tasks with only PNG (no JSON) are treated as incomplete and will re-run")
            else:
                print(f"✓ All tasks completed, nothing to run")
                sys.exit(0)
        else:
            print(f"✓ No completed tasks found, will run all")
            print(f"💡 Tasks with only PNG (no JSON) will be re-run")
    
    if not task_ids:
        print("❌ No tasks to run")
        sys.exit(0)
    
    if args.sequential:
        print(f"\n{'=' * 80}")
        print(f"🚀 Starting sequential benchmark")
        print(f"{'=' * 80}")
        print(f"Total tasks: {len(task_ids)}")
        print(f"Mode: sequential (one by one)")
        print(f"Config: {config_path}")
        if args.headless:
            print("Headless: enabled (auto mode)")
        print(f"{'=' * 80}\n")
        
        actual_config_path, temp_config_file, headless_mode = prepare_runtime_config(
            config_path=config_path,
            headless=args.headless,
        )
        if args.headless:
            if headless_mode == "x_display":
                runtime_display = os.environ.get("DISPLAY", "").strip()
                print(f"🖥️  Headless mode: using X display path ({runtime_display or 'configured x_display'})")
            elif headless_mode == "cloudrendering":
                print("🖥️  Headless mode: using CloudRendering fallback (no DISPLAY detected)")
            print(f"✓ Temp config created: {actual_config_path}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_output_dir = os.path.join(
            args.output_dir,
            benchmark_run_dirname("benchmark_sequential", timestamp, args.save_name),
        )
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        failed_logs_dir = os.path.join(benchmark_output_dir, "failed_logs")
        os.makedirs(failed_logs_dir, exist_ok=True)
        task_logs_dir = Path(benchmark_output_dir) / "task_logs"
        task_logs_dir.mkdir(parents=True, exist_ok=True)
        outputs_completed_path = Path(args.outputs_completed_dir)
        outputs_completed_path.mkdir(parents=True, exist_ok=True)
        csv_lock = Lock()
        
        successful = 0
        failed_model = 0
        failed_external = 0
        copied_to_completed = 0
        
        task_records = []

        if HAS_TQDM:
            task_iterator = tqdm(task_ids, desc="Tasks", unit="task", ncols=100, 
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            task_iterator = task_ids
        
        for idx, task_id in enumerate(task_iterator, 1):
            if HAS_TQDM:
                task_iterator.set_description(f"Task: {task_id}")
            else:
                print(f"\n{'=' * 80}")
                print(f"📋 Task {idx}/{len(task_ids)}: {task_id}")
                print(f"{'=' * 80}")
            
            task_success = False
            task_start_time = time.time()
            task_log_content = []
            task_status = "failed"
            
            task_output_dir = os.path.join(benchmark_output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                "-m",
                "scripts.procthor.work.run_task",
                "--config", str(actual_config_path),
                "--tasks", task_id,
                "--output-dir", task_output_dir
            ]
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
                    timeout=None
                )
                execution_duration = time.time() - execution_start_time
                
                output_lines = []
                if result.stdout:
                    output_lines.extend(result.stdout.splitlines())
                    task_log_content.append(f"=== STDOUT ===\n")
                    task_log_content.append("\n".join(result.stdout.splitlines()))
                    task_log_content.append("\n")
                if result.stderr:
                    output_lines.extend(result.stderr.splitlines())
                    task_log_content.append(f"=== STDERR ===\n")
                    task_log_content.append("\n".join(result.stderr.splitlines()))
                    task_log_content.append("\n")
                
                task_log_content.append(f"=== Run info ===\n")
                task_log_content.append(f"Exit code: {result.returncode}\n")
                task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
                task_log_content.append(f"{'=' * 80}\n\n")
                
                task_log = "\n".join(task_log_content)
                if check_task_success(task_id, task_output_dir, str(actual_config_path), args.headless):
                    task_success = True
                    task_status = "success"
                    successful += 1
                    
                    save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                    if HAS_TQDM:
                        tqdm.write(f"  ✅ Task {task_id} success (evaluate passed)")
                    else:
                        print(f"  ✅ Task {task_id} success (evaluate passed)")
                    
                    if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                        copied_to_completed += 1
                    if update_csv_task_status(csv_path, task_id, 'true', csv_lock):
                        if HAS_TQDM:
                            tqdm.write(f"  ✓ CSV updated: {task_id} -> true")
                        else:
                            print(f"  ✓ CSV updated: {task_id} -> true")
                else:
                    result_json = find_result_json(task_output_dir)
                    failure_reason = determine_failure_reason(result_json_path=result_json)
                    if result_json:
                        if failure_reason == "api_error":
                            task_status = "failed_external"
                            failed_external += 1
                            status_msg = "API error (external)"
                        elif failure_reason == "env_error":
                            task_status = "failed_external"
                            failed_external += 1
                            status_msg = "Env crash (external)"
                        elif failure_reason == "parse_error":
                            task_status = "failed_model"
                            failed_model += 1
                            status_msg = "Parse error (model)"
                        elif failure_reason == "action_error":
                            task_status = "failed_model"
                            failed_model += 1
                            status_msg = "Invalid action (model)"
                        else:
                            task_status = "failed_model"
                            failed_model += 1
                            status_msg = "Evaluate failed after run"
                        
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        save_failed_log(task_id, task_output_dir, failed_logs_dir, 1, output_lines)
                        if HAS_TQDM:
                            tqdm.write(f"  ⚠️  Task {task_id} {status_msg}")
                        else:
                            print(f"  ⚠️  Task {task_id} {status_msg}")
                        if failure_reason not in ("api_error", "env_error"):
                            if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                                copied_to_completed += 1
                            if update_csv_task_status(csv_path, task_id, 'false', csv_lock):
                                if HAS_TQDM:
                                    tqdm.write(f"  ✓ CSV updated: {task_id} -> false")
                                else:
                                    print(f"  ✓ CSV updated: {task_id} -> false")
                    else:
                        task_status = "failed_external"
                        failed_external += 1
                        status_msg = "No JSON produced (external)"
                        
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        save_failed_log(task_id, task_output_dir, failed_logs_dir, 1, output_lines)
                        
                        if HAS_TQDM:
                            tqdm.write(f"  ⚠️  Task {task_id} {status_msg}")
                        else:
                            print(f"  ⚠️  Task {task_id} {status_msg}")

            except KeyboardInterrupt:
                if HAS_TQDM:
                    tqdm.write(f"\n⚠️ User interrupt, ran {idx-1}/{len(task_ids)} tasks")
                else:
                    print(f"\n⚠️ User interrupt, ran {idx-1}/{len(task_ids)} tasks")
                task_log_content.append(f"\n⚠️ User interrupt\n")
                task_log = "\n".join(task_log_content)
                save_task_log(task_id, task_log, task_logs_dir, 1, "interrupted")
                failed_external += 1
                task_status = "interrupted"
                raise
            except Exception as e:
                execution_duration = time.time() - execution_start_time if 'execution_start_time' in locals() else 0
                error_msg = str(e)
                if HAS_TQDM:
                    tqdm.write(f"  ❌ Task {task_id} exception: {e}")
                else:
                    print(f"  ❌ Task {task_id} exception: {e}")
                
                task_log_content.append(f"=== Exception ===\n")
                task_log_content.append(f"Error: {error_msg}\n")
                task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
                task_log_content.append(f"{'=' * 80}\n\n")
                
                task_log = "\n".join(task_log_content)
                save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                save_failed_log(task_id, task_output_dir, failed_logs_dir, 1, [error_msg])
                result_json = find_result_json(task_output_dir)
                failure_reason = determine_failure_reason(result_json_path=result_json)
                
                if failure_reason in ("parse_error", "action_error"):
                    failed_model += 1
                    task_status = "failed_model"
                    if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                        copied_to_completed += 1
                    update_csv_task_status(csv_path, task_id, 'false', csv_lock)
                else:
                    failed_external += 1
                    task_status = "failed_external"
            
            task_duration = time.time() - task_start_time
            task_records.append({
                'task_id': task_id,
                'status': task_status,
                'attempts': 1,
                'duration': task_duration,
                'success': task_success,
                'log_path': None
            })
            
            if HAS_TQDM:
                task_iterator.set_postfix({
                    'ok': successful,
                    'model': failed_model,
                    'external': failed_external,
                    'copied': copied_to_completed
                })
        
        if temp_config_file and os.path.exists(temp_config_file.name):
            try:
                os.unlink(temp_config_file.name)
            except Exception as e:
                print(f"⚠️ Failed to remove temp config: {e}")
        
        summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
        total_duration = sum(record['duration'] for record in task_records)
        avg_duration = total_duration / len(task_records) if task_records else 0
        
        with open(summary_log_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Benchmark run summary\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV: {csv_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Output dir: {benchmark_output_dir}\n")
            f.write(f"Mode: sequential\n")
            if args.headless:
                f.write(f"Headless: enabled\n")
            if args.max_steps:
                f.write(f"Max steps: {args.max_steps}\n")
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Summary\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total tasks: {len(task_ids)}\n")
            f.write(f"Success: {successful} ({successful/len(task_ids)*100:.1f}%)\n")
            f.write(f"Model failure: {failed_model} ({failed_model/len(task_ids)*100:.1f}%)\n")
            f.write(f"External failure: {failed_external} ({failed_external/len(task_ids)*100:.1f}%)\n")
            f.write(f"Copied to {args.outputs_completed_dir}: {copied_to_completed}\n")
            f.write(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)\n")
            f.write(f"Avg time: {avg_duration:.2f}s\n")
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Task details\n")
            f.write(f"{'=' * 80}\n\n")
        
            successful_tasks = [r for r in task_records if r['success']]
            failed_tasks = [r for r in task_records if not r['success']]
        
            if successful_tasks:
                f.write(f"Success ({len(successful_tasks)}):\n")
                f.write(f"{'-' * 80}\n")
                for record in successful_tasks:
                    f.write(f"  ✅ {record['task_id']}: {record['duration']:.2f}s, {record['attempts']} attempt(s)\n")
                f.write(f"\n")
        
            if failed_tasks:
                f.write(f"Failed ({len(failed_tasks)}):\n")
                f.write(f"{'-' * 80}\n")
                for record in failed_tasks:
                    f.write(f"  ❌ {record['task_id']}: {record['duration']:.2f}s, {record['attempts']} attempt(s)\n")
                f.write(f"\n")
        
            f.write(f"{'=' * 80}\n")
            f.write(f"All tasks\n")
            f.write(f"{'=' * 80}\n")
            for i, record in enumerate(task_records, 1):
                status_icon = "✅" if record['success'] else "❌"
                f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                       f"status: {record['status']:10s} | "
                       f"attempts: {record['attempts']:2d} | "
                       f"duration: {record['duration']:8.2f}s")
                if record['log_path']:
                    f.write(f" | log: {record['log_path']}")
                f.write(f"\n")
        
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Log locations\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Task logs: {task_logs_dir}\n")
            f.write(f"Summary: {summary_log_path}\n")
            if failed_model + failed_external > 0:
                f.write(f"Failed logs: {failed_logs_dir}\n")
        
        csv_stats = count_csv_status(csv_path)
        print(f"\n{'=' * 80}")
        print(f"🎉 Sequential run complete")
        print(f"{'=' * 80}")
        print(f"Total tasks: {len(task_ids)}")
        print(f"Success: {successful}")
        print(f"Model failure: {failed_model}")
        print(f"External failure: {failed_external}")
        print(f"Copied to {args.outputs_completed_dir}: {copied_to_completed}")
        print(f"Output dir: {benchmark_output_dir}")
        print(f"Task logs: {task_logs_dir}")
        print(f"Summary log: {summary_log_path}")
        if failed_model + failed_external > 0:
            print(f"Failed logs: {failed_logs_dir}")
        
        print(f"\n{'=' * 80}")
        print(f"📊 CSV status")
        print(f"{'=' * 80}")
        total_csv = csv_stats["total"]
        if total_csv > 0:
            true_count = csv_stats["true"]
            false_count = csv_stats["false"]
            null_count = csv_stats["null"]
            print(f"Total: {total_csv}")
            print(f"true: {true_count} ({true_count/total_csv*100:.1f}%)")
            print(f"false: {false_count} ({false_count/total_csv*100:.1f}%)")
            print(f"null: {null_count} ({null_count/total_csv*100:.1f}%)")
        
        if failed_external > 0:
            print(f"\n{'=' * 80}")
            print(f"📋 Analyze null (external) reasons")
            print(f"{'=' * 80}")
            try:
                import importlib.util
                analyze_errors_path = Path("analyze_errors.py")
                if analyze_errors_path.exists():
                    spec = importlib.util.spec_from_file_location("analyze_errors", analyze_errors_path)
                    analyze_errors = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(analyze_errors)
                    analyze_errors.analyze_benchmark_errors(str(benchmark_output_dir))
                else:
                    print(f"⚠️  analyze_errors.py not found, skip")
            except Exception as e:
                print(f"⚠️  analyze_errors.py failed: {e}")
        
        print(f"{'=' * 80}\n")
        
        sys.exit(0 if (failed_model + failed_external) == 0 else 1)
    
    actual_config_path, temp_config_file, headless_mode = prepare_runtime_config(
        config_path=config_path,
        headless=args.headless,
    )
    if args.headless:
        if headless_mode == "x_display":
            runtime_display = os.environ.get("DISPLAY", "").strip()
            print(f"🖥️  Headless mode: using X display path ({runtime_display or 'configured x_display'})")
        elif headless_mode == "cloudrendering":
            print("🖥️  Headless mode: using CloudRendering fallback (no DISPLAY detected)")
        print(f"✓ Temp config created: {actual_config_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_dir = os.path.join(
        args.output_dir,
        benchmark_run_dirname("benchmark", timestamp, args.save_name),
    )
    os.makedirs(benchmark_output_dir, exist_ok=True)
    task_logs_dir = Path(benchmark_output_dir) / "task_logs"
    task_logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_completed_path = Path(args.outputs_completed_dir)
    outputs_completed_path.mkdir(parents=True, exist_ok=True)
    csv_lock = Lock()
    
    print(f"\n{'=' * 80}")
    print(f"🚀 Starting parallel benchmark")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Workers: {args.workers}")
    print(f"Config: {actual_config_path}")
    print(f"Output dir: {benchmark_output_dir}")
    if args.headless:
        print("Headless: enabled (auto mode)")
    print(f"{'=' * 80}\n")
    
    def execute_task(task_id: str) -> dict:
        """Run one task; on success copy to outputs_completed and update CSV."""
        task_start_time = time.time()
        task_log_content = []
        task_status = "failed"
        task_success = False
        task_output_dir = os.path.join(benchmark_output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        cmd = [
            sys.executable,
            "-m",
                "scripts.procthor.work.run_task",
            "--config", str(actual_config_path),
            "--tasks", task_id,
            "--output-dir", task_output_dir
        ]
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
                timeout=None
            )
            execution_duration = time.time() - execution_start_time
            
            if result.stdout:
                task_log_content.append(f"=== STDOUT ===\n")
                task_log_content.append(result.stdout)
                task_log_content.append("\n")
            if result.stderr:
                task_log_content.append(f"=== STDERR ===\n")
                task_log_content.append(result.stderr)
                task_log_content.append("\n")
            
            task_log_content.append(f"=== Run info ===\n")
            task_log_content.append(f"Exit code: {result.returncode}\n")
            task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_content.append(f"{'=' * 80}\n\n")
            
            task_log = "".join(task_log_content)
            if check_task_success(task_id, task_output_dir, str(actual_config_path), args.headless):
                task_success = True
                task_status = "success"
                save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                    update_csv_task_status(csv_path, task_id, 'true', csv_lock)
                    print(f"  ✅ {task_id} success (evaluate passed) -> copied and CSV updated")
            else:
                result_json = find_result_json(task_output_dir)
                failure_reason = determine_failure_reason(result_json_path=result_json)
                if result_json:
                    if failure_reason == "api_error":
                        task_status = "failed_external"
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        print(f"  ⚠️  {task_id} API error -> null")
                    elif failure_reason == "env_error":
                        task_status = "failed_external"
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        print(f"  ⚠️  {task_id} Env crash -> null")
                    elif failure_reason == "parse_error":
                        task_status = "failed_model"
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                            update_csv_task_status(csv_path, task_id, 'false', csv_lock)
                            print(f"  ❌ {task_id} Parse error -> false")
                    elif failure_reason == "action_error":
                        task_status = "failed_model"
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                            update_csv_task_status(csv_path, task_id, 'false', csv_lock)
                            print(f"  ❌ {task_id} Invalid action -> false")
                    else:
                        task_status = "failed_model"
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                            update_csv_task_status(csv_path, task_id, 'false', csv_lock)
                            print(f"  ❌ {task_id} Evaluate failed -> false")
                else:
                    task_status = "failed_external"
                    save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                    print(f"  ⚠️  {task_id} No JSON -> null")
        except KeyboardInterrupt:
            task_status = "interrupted"
            task_log = "".join(task_log_content) if task_log_content else ""
            save_task_log(task_id, task_log, task_logs_dir, 1, "interrupted")
        except Exception as e:
            execution_duration = time.time() - execution_start_time if 'execution_start_time' in locals() else 0
            task_log_content.append(f"=== Exception ===\n")
            task_log_content.append(f"Error: {str(e)}\n")
            task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_content.append(f"{'=' * 80}\n\n")
            
            task_log = "".join(task_log_content)
            save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
            result_json = find_result_json(task_output_dir)
            failure_reason = determine_failure_reason(result_json_path=result_json)
            if failure_reason in ("parse_error", "action_error"):
                task_status = "failed_model"
                if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                    update_csv_task_status(csv_path, task_id, 'false', csv_lock)
                    print(f"  ❌ {task_id} Model failure -> false")
            else:
                task_status = "failed_external"
                print(f"  ⚠️  {task_id} External -> null")

        task_duration = time.time() - task_start_time
        return {
            'task_id': task_id,
            'status': task_status,
            'attempts': 1,
            'duration': task_duration,
            'success': task_success,
            'log_path': None
        }
    
    task_records = []
    successful = 0
    failed_model = 0
    failed_external = 0
    copied_to_completed = 0
    exit_code = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(execute_task, task_id): task_id
                for task_id in task_ids
            }
            
            if HAS_TQDM:
                task_iterator = tqdm(as_completed(future_to_task), total=len(task_ids),
                                    desc="Tasks", unit="task", ncols=100,
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                task_iterator = as_completed(future_to_task)
            
            for future in task_iterator:
                try:
                    result = future.result()
                    task_records.append(result)
                    
                    if result['success']:
                        successful += 1
                        copied_to_completed += 1
                    else:
                        if result['status'] == "failed_model":
                            failed_model += 1
                            copied_to_completed += 1
                        else:
                            failed_external += 1
                    
                    if HAS_TQDM:
                        task_iterator.set_postfix({
                            'ok': successful,
                            'model': failed_model,
                            'external': failed_external,
                            'copied': copied_to_completed
                        })
                except Exception as e:
                    print(f"\n❌ Task exception: {e}")
                    failed_external += 1
    
    except KeyboardInterrupt:
        print("\n⚠️ User interrupt")
        exit_code = 1
    else:
        exit_code = 0 if (failed_model + failed_external) == 0 else 1
    
    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    total_duration = sum(record['duration'] for record in task_records)
    avg_duration = total_duration / len(task_records) if task_records else 0
    
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"Benchmark run summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Output dir: {benchmark_output_dir}\n")
        f.write(f"Mode: parallel (workers: {args.workers})\n")
        if args.headless:
            f.write(f"Headless: enabled\n")
        if args.max_steps:
            f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Total tasks: {len(task_ids)}\n")
        f.write(f"Success: {successful} ({successful/len(task_ids)*100:.1f}%)\n")
        f.write(f"Model failure: {failed_model} ({failed_model/len(task_ids)*100:.1f}%)\n")
        f.write(f"External failure: {failed_external} ({failed_external/len(task_ids)*100:.1f}%)\n")
        f.write(f"Copied to {args.outputs_completed_dir}: {copied_to_completed}\n")
        f.write(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)\n")
        f.write(f"Avg time: {avg_duration:.2f}s\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Task details\n")
        f.write(f"{'=' * 80}\n\n")
        
        successful_tasks = [r for r in task_records if r['success']]
        failed_tasks = [r for r in task_records if not r['success']]
        
        if successful_tasks:
            f.write(f"Success ({len(successful_tasks)}):\n")
            f.write(f"{'-' * 80}\n")
            for record in successful_tasks:
                f.write(f"  ✅ {record['task_id']}: {record['duration']:.2f}s\n")
            f.write(f"\n")
        
        if failed_tasks:
            f.write(f"Failed ({len(failed_tasks)}):\n")
            f.write(f"{'-' * 80}\n")
            for record in failed_tasks:
                f.write(f"  ❌ {record['task_id']}: {record['duration']:.2f}s\n")
            f.write(f"\n")
        
        f.write(f"{'=' * 80}\n")
        f.write(f"All tasks\n")
        f.write(f"{'=' * 80}\n")
        for i, record in enumerate(task_records, 1):
            status_icon = "✅" if record['success'] else "❌"
            f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                   f"status: {record['status']:10s} | "
                   f"duration: {record['duration']:8.2f}s")
            if record['log_path']:
                f.write(f" | log: {record['log_path']}")
            f.write(f"\n")
        
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Log locations\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Task logs: {task_logs_dir}\n")
        f.write(f"Summary: {summary_log_path}\n")
    
    csv_stats = count_csv_status(csv_path)
    print(f"\n{'=' * 80}")
    print(f"🎉 Parallel run complete")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Success: {successful}")
    print(f"Model failure: {failed_model}")
    print(f"External failure: {failed_external}")
    print(f"Copied to {args.outputs_completed_dir}: {copied_to_completed}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Task logs: {task_logs_dir}")
    print(f"Summary log: {summary_log_path}")
    
    print(f"\n{'=' * 80}")
    print(f"📊 CSV status")
    print(f"{'=' * 80}")
    total_csv = csv_stats["total"]
    if total_csv > 0:
        true_count = csv_stats["true"]
        false_count = csv_stats["false"]
        null_count = csv_stats["null"]
        print(f"Total: {total_csv}")
        print(f"true: {true_count} ({true_count/total_csv*100:.1f}%)")
        print(f"false: {false_count} ({false_count/total_csv*100:.1f}%)")
        print(f"null: {null_count} ({null_count/total_csv*100:.1f}%)")
    
    if failed_external > 0:
        print(f"\n{'=' * 80}")
        print(f"📋 Analyze null (external) reasons")
        print(f"{'=' * 80}")
        try:
            import importlib.util
            analyze_errors_path = Path("analyze_errors.py")
            if analyze_errors_path.exists():
                spec = importlib.util.spec_from_file_location("analyze_errors", analyze_errors_path)
                analyze_errors = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(analyze_errors)
                analyze_errors.analyze_benchmark_errors(str(benchmark_output_dir))
            else:
                print(f"⚠️  analyze_errors.py not found, skip")
        except Exception as e:
            print(f"⚠️  analyze_errors.py failed: {e}")

    print(f"{'=' * 80}\n")

    if temp_config_file and os.path.exists(temp_config_file.name):
        try:
            os.unlink(temp_config_file.name)
            print(f"✓ Temp config removed")
        except Exception as e:
            print(f"⚠️ Failed to remove temp config: {e}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
