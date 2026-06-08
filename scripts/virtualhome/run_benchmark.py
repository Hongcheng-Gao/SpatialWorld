#!/usr/bin/env python3
"""
Read task IDs from CSV and run benchmark (parallel or sequential).

NOTE: Global task-level retry is disabled for fair model evaluation.
All retries are step-level:
- Parse errors: retry 3x in think_node
- API errors (400/500/timeout): retry 3x + backoff in provider.py
- Env execution errors: retry 3x in act_node

If step-level retries are exhausted, the case is marked failed (no full-task retry).

CLI     :
- --csv:    CSV   （  ），      task_id，Completed/Status       true/false/null 
- --workers:    worker  （       ） 
- --config:        （   experiments/configs/virtualhome/config_close_gpt-5.yaml） 
- --max-steps:      summary，      python -m scripts.virtualhome.work.run_task（    ，        ） 
- --output-dir:    benchmark      ；        （  --save-name） 
- --save-name:         ；        {save-name}_{   }，      benchmark_{   }   benchmark_sequential_{   } 
- --headless:   CloudRendering     ，      config   env.platform 
- --sequential:       （     ，   API       ） 
- --skip-completed:        （              JSON   ） 
- --outputs-completed-dir:          （success   model failure    ，    ） 
- --task:      ，      task_id；       CSV      null     

      :
- success -> true
- false（    ）：
  * failure_type = parse_error/action_error/model_error
  * fail_reason   ：Model claimed DONE but（DONE   ） Consecutive X action failures（  4   ） maximum step/max_steps（      ） Model determined task cannot（    FAIL）
- null（    ，   ）：
  * failure_type = api_error/env_error/external_error（   task_result=failure     null）
  *       
-              JSON，    run_error.txt    return code   stdout/stderr    
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
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mllm_base_agent.environments.virtualhome.backend_utils import (
    DEFAULT_BACKEND_ARGS,
    DEFAULT_BACKEND_EXE,
    DEFAULT_STARTUP_TIMEOUT,
    is_port_open,
    launch_backend,
    terminate_backend,
    wait_for_port,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed, using simple progress. Install: pip install tqdm")


# Global locks for cross-thread safety in parallel mode
EVAL_LOCK = Lock()
INFLIGHT_TASKS_LOCK = Lock()
INFLIGHT_TASKS = set()
EXTERNAL_FAILURE_TYPES = {"api_error", "env_error", "external_error"}
MODEL_FAILURE_TYPES = {"parse_error", "action_error", "model_error"}


class _SkipSequential(Exception):
    """Internal sentinel to skip sequential branch cleanly."""


def _enable_console_write_fallback() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _child_process_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


_enable_console_write_fallback()


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


def find_completed_tasks(output_dir: str, save_name: Optional[str] = None) -> set:
    """
    Find completed tasks (log.json or episode_*.json in latest benchmark dirs).

    Args:
        output_dir: Output root dir.
        save_name:    （  --save-name），     {save_name}_*            

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
    custom_dirs = []
    if save_name:
        custom_dirs = sorted(
            [d for d in output_path.glob(f"{save_name}_*") if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )
    all_dirs = benchmark_dirs + sequential_dirs + custom_dirs

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
            task_id_match = re.match(
                r"^((?:ai2thor|carla|virtualhome)\d+)(?:_retry_\d+)?$",
                task_name_with_retry,
            )
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


def infer_env_type(task_id: str, config_path: str) -> str:
    """Infer environment type from task id first, config as fallback."""
    normalized_task_id = normalize_task_id(task_id).lower()
    for prefix in ("virtualhome", "ai2thor", "carla"):
        if normalized_task_id.startswith(prefix):
            return prefix

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        env_type = str(config_data.get("env", {}).get("type", "ai2thor")).lower()
        if env_type in {"virtualhome", "ai2thor", "carla"}:
            return env_type
    except Exception:
        pass

    return "ai2thor"


def infer_env_type_from_config(config_path: str) -> str:
    return infer_env_type("", config_path)


def get_virtualhome_endpoint(config_path: Path) -> Tuple[str, int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        env_cfg = config_data.get("env", {})
        host = str(env_cfg.get("url", "127.0.0.1"))
        try:
            port = int(str(env_cfg.get("port", "8080")))
        except (TypeError, ValueError):
            port = 8080
        return host, port
    except Exception:
        return "127.0.0.1", 8080


def _find_eval_result_json(eval_output_root: Path, env_type: str) -> Optional[Path]:
    """Find evaluation result json for a specific environment."""
    if env_type == "virtualhome":
        eval_dirs = sorted(eval_output_root.glob("vh_eval_*"))
        if not eval_dirs:
            return None
        result_files = sorted(eval_dirs[-1].glob("vh_action_result_*.json"))
        return result_files[-1] if result_files else None

    if env_type == "carla":
        eval_dirs = sorted(eval_output_root.glob("eval_*"))
        if not eval_dirs:
            return None
        result_file = eval_dirs[-1] / "result.json"
        return result_file if result_file.exists() else None

    eval_dirs = sorted(eval_output_root.glob("action_eval_*"))
    if not eval_dirs:
        return None
    result_files = sorted(eval_dirs[-1].glob("action_sequence_result_*.json"))
    return result_files[-1] if result_files else None


def evaluate_task(task_id: str, config_path: str, headless: bool = False) -> bool:
    """
    Run evaluate_action_sequence.py to evaluate task success.

    Returns:
        True if evaluation succeeds.
    """
    eval_output_root = None
    try:
        with EVAL_LOCK:
            env_type = infer_env_type(task_id, config_path)
            if env_type == "virtualhome":
                eval_script = "scripts/virtualhome/evaluate_action_sequence.py"
            elif env_type == "carla":
                eval_script = "scripts/carla/evaluate_action_sequence.py"
            else:
                eval_script = "scripts/ai2thor/evaluate_action_sequence.py"

            os.makedirs("outputs", exist_ok=True)
            eval_output_root = Path(
                tempfile.mkdtemp(prefix=f"{env_type}_eval_", dir="outputs")
            )
            cmd = [
                sys.executable,
                eval_script,
                "--task", task_id,
                "--config", config_path,
                "--output-dir", str(eval_output_root),
            ]

            if headless and env_type == "ai2thor":
                cmd.append("--headless")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_child_process_env(),
                timeout=300,
            )

            if result.returncode != 0:
                print(f"  ⚠️  Evaluate command failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:500]}")
                return False

            result_json = _find_eval_result_json(eval_output_root, env_type)
            if not result_json:
                print(f"  ⚠️  No evaluate result JSON found")
                return False

            with open(result_json, 'r', encoding='utf-8') as f:
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
    finally:
        if eval_output_root and eval_output_root.exists():
            shutil.rmtree(eval_output_root, ignore_errors=True)


def check_task_success(task_id: str, task_output_dir: str, config_path: str, headless: bool = False) -> bool:
    """
    Check if task completed successfully (JSON present, then evaluate).

    Returns:
        True if task completed and evaluate passes.
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

        elif result_json.name == 'log.json':
            metadata = result_data.get('metadata', {})
            task_result = metadata.get('task_result', '')
            failure_type = metadata.get("failure_type")
            if task_result == 'failure':
                fail_reason = metadata.get('fail_reason', 'Unknown reason')
                print(f"  ⚠️  Task run failed: {fail_reason}")
                return False
            if failure_type in EXTERNAL_FAILURE_TYPES:
                fail_reason = metadata.get('fail_reason', 'External error')
                print(f"  ⚠️  Task run external failure: {fail_reason}")
                return False
            if task_result is None and failure_type:
                fail_reason = metadata.get('fail_reason', 'Unknown reason')
                print(f"  ⚠️  Task run not completed: {fail_reason}")
                return False
    except Exception as e:
        print(f"  ⚠️  Failed to read task result JSON: {e}")

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
    """Normalize task ID (e.g. ai2thor_04000 -> ai2thor04000)."""
    for prefix in ("ai2thor", "carla", "virtualhome"):
        underscored = f"{prefix}_"
        if '_' in task_id and task_id.startswith(underscored):
            return task_id.replace(underscored, prefix, 1)
    return task_id


def determine_failure_reason(task_log_content: str = None, result_json_path: Path = None) -> str:
    """
    Determine failure reason; prefer failure_type from JSON.

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


def update_csv_task_record(
    csv_path: Path,
    task_id: str,
    status: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    lock: Lock = None,
) -> bool:
    """Update CSV row for a task: Completed status + optional extra analysis fields.

    extra_fields keys example:
      - golden_action
      - instruction
      - token_total
      - failure_reason
      - actual_actions
      - golden_actions_count
      - actual_actions_count
    """
    if lock:
        lock.acquire()

    try:
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        if not rows:
            return False

        header = rows[0]

        def ensure_column(col_name: str) -> int:
            for idx, col in enumerate(header):
                if col.strip().lower() == col_name.strip().lower():
                    return idx
            header.append(col_name)
            new_idx = len(header) - 1
            for row_i in range(1, len(rows)):
                if len(rows[row_i]) <= new_idx:
                    rows[row_i].extend([''] * (new_idx + 1 - len(rows[row_i])))
            return new_idx

        completed_col_idx = None
        for idx, col in enumerate(header):
            if col.strip().lower() in ['completed', 'status']:
                completed_col_idx = idx
                break
        if completed_col_idx is None:
            completed_col_idx = ensure_column("Completed")

        extra_idx_map = {}
        if extra_fields:
            for field_name in extra_fields.keys():
                extra_idx_map[field_name] = ensure_column(field_name)

        normalized_task_id = normalize_task_id(task_id)
        updated = False

        for i in range(1, len(rows)):
            if not rows[i] or not rows[i][0].strip():
                continue
            csv_task_id = rows[i][0].strip()
            normalized_csv_id = normalize_task_id(csv_task_id)
            if normalized_csv_id != normalized_task_id and csv_task_id != task_id:
                continue

            max_needed = max(
                [completed_col_idx] + (list(extra_idx_map.values()) if extra_idx_map else [completed_col_idx])
            )
            while len(rows[i]) <= max_needed:
                rows[i].append('')

            if status is not None:
                old_status = rows[i][completed_col_idx].strip().lower() if rows[i][completed_col_idx] else ''
                if status.lower() == 'null':
                    new_status = ''
                else:
                    new_status = status
                if old_status != new_status.strip().lower():
                    rows[i][completed_col_idx] = new_status
                    updated = True

            if extra_fields:
                for field_name, field_value in extra_fields.items():
                    idx = extra_idx_map[field_name]
                    new_value = '' if field_value is None else str(field_value)
                    if rows[i][idx] != new_value:
                        rows[i][idx] = new_value
                        updated = True

            break

        if updated:
            backup_path = csv_path.with_suffix('.csv.backup')
            if not backup_path.exists():
                shutil.copy2(csv_path, backup_path)

            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        return updated

    except Exception as e:
        print(f"  ⚠️  Update CSV record failed ({task_id}): {e}")
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


def deduplicate_task_ids(task_ids: list) -> Tuple[list, list]:
    """Deduplicate tasks by normalized task ID. Returns (unique_task_ids, skipped_duplicates)."""
    unique_task_ids = []
    seen = set()
    skipped = []

    for task_id in task_ids:
        normalized = normalize_task_id(task_id.strip())
        if normalized in seen:
            skipped.append(task_id)
            continue
        seen.add(normalized)
        unique_task_ids.append(task_id)

    return unique_task_ids, skipped


def load_task_action_count(task_id: str) -> Optional[int]:
    """Load golden action count from tasks/<task_id>/task.json."""
    normalized = normalize_task_id(task_id)
    task_json = Path(os.environ.get("SPATIAL_TASKS_ROOT", "data/virtualhome/tasks")) / normalized / "task.json"
    if not task_json.exists():
        return None

    try:
        with open(task_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        golden_actions = data.get("golden_actions")
        if isinstance(golden_actions, dict):
            steps = golden_actions.get("steps")
            if isinstance(steps, int):
                return steps

            actions = golden_actions.get("actions")
            if isinstance(actions, list):
                count = 0
                for action in actions:
                    if isinstance(action, str) and action.strip().upper() == "DONE":
                        continue
                    count += 1
                return count
        elif isinstance(golden_actions, list):
            count = 0
            for action in golden_actions:
                if isinstance(action, str) and action.strip().upper() == "DONE":
                    continue
                count += 1
            return count
        elif isinstance(golden_actions, str):
            parts = [x.strip() for x in golden_actions.split(",") if x.strip()]
            return len([x for x in parts if x.upper() != "DONE"])
    except Exception as e:
        print(f"  ⚠️  Failed to load action count for {task_id}: {e}")

    return None


def load_task_metadata(task_id: str) -> Dict[str, Any]:
    """Load task metadata from tasks/<task_id>/task.json.

    Returns keys:
      - instruction: str | None
      - golden_action_count: int | None
      - golden_action_text: str | None
    """
    normalized = normalize_task_id(task_id)
    task_json = Path(os.environ.get("SPATIAL_TASKS_ROOT", "data/virtualhome/tasks")) / normalized / "task.json"
    info = {
        "instruction": None,
        "golden_action_count": None,
        "golden_action_text": None,
    }

    if not task_json.exists():
        return info

    try:
        with open(task_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        info["instruction"] = data.get("instruction") or data.get("task_name")
        golden_actions = data.get("golden_actions")

        actions_list = None
        if isinstance(golden_actions, dict):
            steps = golden_actions.get("steps")
            if isinstance(steps, int):
                info["golden_action_count"] = steps

            actions = golden_actions.get("actions")
            if isinstance(actions, list):
                actions_list = [str(a).strip() for a in actions if str(a).strip()]
                if info["golden_action_count"] is None:
                    info["golden_action_count"] = len([
                        a for a in actions_list if a.upper() != "DONE"
                    ])

        elif isinstance(golden_actions, list):
            actions_list = [str(a).strip() for a in golden_actions if str(a).strip()]
            info["golden_action_count"] = len([
                a for a in actions_list if a.upper() != "DONE"
            ])

        elif isinstance(golden_actions, str):
            actions_list = [x.strip() for x in golden_actions.split(",") if x.strip()]
            info["golden_action_count"] = len([
                a for a in actions_list if a.upper() != "DONE"
            ])

        if actions_list:
            info["golden_action_text"] = " | ".join(actions_list)

    except Exception as e:
        print(f"  ⚠️  Failed to load task metadata for {task_id}: {e}")

    return info


def extract_token_stats_from_text(text: str) -> Dict[str, Optional[int]]:
    """Extract and sum prompt/completion/total tokens from stdout/stderr text."""
    if not text:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    # Supports patterns like:
    # - "prompt_tokens": 123
    # - prompt_tokens=123
    # - prompt_tokens: 123
    # - 'prompt_tokens': 123
    token_value_pattern = r"(?:['\"])?{key}(?:['\"])?\s*[:=]\s*(\d+)"
    prompt_matches = [int(x) for x in re.findall(token_value_pattern.format(key="prompt_tokens"), text, flags=re.IGNORECASE)]
    completion_matches = [int(x) for x in re.findall(token_value_pattern.format(key="completion_tokens"), text, flags=re.IGNORECASE)]
    total_matches = [int(x) for x in re.findall(token_value_pattern.format(key="total_tokens"), text, flags=re.IGNORECASE)]

    prompt_tokens = sum(prompt_matches) if prompt_matches else None
    completion_tokens = sum(completion_matches) if completion_matches else None

    if total_matches:
        total_tokens = sum(total_matches)
    elif prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    else:
        total_tokens = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def extract_token_stats_from_result_json(result_json: Optional[Path]) -> Dict[str, Optional[int]]:
    """Extract token stats from result JSON metadata token_usage."""
    empty = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if not result_json or not result_json.exists():
        return empty

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}

        if not isinstance(token_usage, dict):
            return empty

        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")

        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        except Exception:
            prompt_tokens = None
        try:
            completion_tokens = int(completion_tokens) if completion_tokens is not None else None
        except Exception:
            completion_tokens = None
        try:
            total_tokens = int(total_tokens) if total_tokens is not None else None
        except Exception:
            total_tokens = None

        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    except Exception:
        return empty


def merge_token_stats(primary: Dict[str, Optional[int]], fallback: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """Merge token stats, preferring non-empty primary values."""
    merged = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = primary.get(key) if primary.get(key) is not None else fallback.get(key)

    if merged.get("total_tokens") is None and (
        merged.get("prompt_tokens") is not None or merged.get("completion_tokens") is not None
    ):
        merged["total_tokens"] = (merged.get("prompt_tokens") or 0) + (merged.get("completion_tokens") or 0)

    return merged


def extract_actual_actions(result_json: Optional[Path]) -> Dict[str, Any]:
    """Extract actual executed actions from result JSON.

    Returns keys:
      - actual_action_count: int | None
      - actual_action_text: str | None
    """
    info = {
        "actual_action_count": None,
        "actual_action_text": None,
    }

    if not result_json or not result_json.exists():
        return info

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        actions = []

        if isinstance(data, dict) and result_json.name == "log.json":
            messages = data.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    action_executed = msg.get("action_executed")
                    if action_executed:
                        actions.append(str(action_executed).strip())

        if not actions and isinstance(data, dict) and result_json.name.startswith("episode_"):
            trajectory = data.get("trajectory", [])
            if isinstance(trajectory, list):
                for step in trajectory:
                    if not isinstance(step, dict):
                        continue
                    action_string = step.get("action_string")
                    if action_string:
                        actions.append(str(action_string).strip())

            if not actions:
                action_sequence = data.get("action_sequence")
                if isinstance(action_sequence, str) and action_sequence.strip():
                    actions = [x.strip() for x in action_sequence.split("->") if x.strip()]

        if actions:
            info["actual_action_count"] = len(actions)
            info["actual_action_text"] = " | ".join(actions)

    except Exception as e:
        print(f"  ⚠️  Failed to extract actual actions from {result_json}: {e}")

    return info


def build_csv_extra_fields(
    task_metadata: Dict[str, Any],
    actual_actions_info: Dict[str, Any],
    token_stats: Dict[str, Optional[int]],
    failure_reason: Optional[str],
) -> Dict[str, Any]:
    """Build extra CSV columns for per-task analysis."""
    return {
        "golden_action": task_metadata.get("golden_action_text"),
        "instruction": task_metadata.get("instruction"),
        "token_total": token_stats.get("total_tokens"),
        "failure_reason": failure_reason,
        "actual_actions": actual_actions_info.get("actual_action_text"),
        "golden_actions_count": task_metadata.get("golden_action_count"),
        "actual_actions_count": actual_actions_info.get("actual_action_count"),
    }


def read_result_status_info(result_json: Optional[Path]) -> Dict[str, Any]:
    """Read structured result from result json.

    Returns keys:
      - has_result_json: bool
      - task_result: 'success' | 'failure' | None
      - failure_type: str | None
      - fail_reason: str | None
    """
    info = {
        "has_result_json": bool(result_json and result_json.exists()),
        "task_result": None,
        "failure_type": None,
        "fail_reason": None,
    }

    if not result_json or not result_json.exists():
        return info

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        task_result = metadata.get("task_result")
        if task_result in ("success", "failure"):
            info["task_result"] = task_result
        elif isinstance(data, dict) and result_json.name.startswith("episode_"):
            success = data.get("success")
            if success is True:
                info["task_result"] = "success"
            elif success is False:
                info["task_result"] = "failure"

        failure_type = data.get("failure_type") if isinstance(data, dict) else None
        if not failure_type:
            failure_type = metadata.get("failure_type")
        info["failure_type"] = failure_type

        fail_reason = metadata.get("fail_reason")
        if not fail_reason and isinstance(data, dict):
            fail_reason = data.get("fail_reason")
        info["fail_reason"] = fail_reason

    except Exception as e:
        info["fail_reason"] = f"Failed to parse result JSON: {e}"

    return info


def decide_csv_status_from_result(result_info: Dict[str, Any], fallback_failure_type: str) -> Optional[str]:
    """Decide CSV Completed status, considering both failure_type and fail_reason.

    Rules:
    - api_error/env_error/external_error → null (external issues, can retry)
    - parse_error/action_error/model_error → false (model faults)
    - Specific fail_reason patterns → false (model behavior issues):
      * "Model claimed DONE but" (DONE but wrong)
      * "Consecutive" + "action failures" (4 consecutive failures)
      * "maximum step" or "max_steps" (exceeded max steps)
      * "Model determined task cannot" (model output FAIL)
    - task_result == "success" → true
    - Other cases → null

    Returns:
      - 'true' / 'false' / None (None means keep null)
    """
    task_result = result_info.get("task_result")
    failure_type = result_info.get("failure_type") or fallback_failure_type
    fail_reason = result_info.get("fail_reason", "")

    # Success case
    if task_result == "success":
        return "true"

    # First, use fail_reason patterns since failure_type may be missing (null) and fallback
    # can incorrectly classify it as external_error. If fail_reason clearly indicates
    # model behavior issues, prefer "false".
    if fail_reason:
        fail_reason_lower = fail_reason.lower()
        # Model claimed DONE but success conditions not met
        if "model claimed done" in fail_reason_lower or "done but" in fail_reason_lower:
            return "false"
        # Consecutive action failures (early stop)
        if "consecutive" in fail_reason_lower and "action failures" in fail_reason_lower:
            return "false"
        # Exceeded maximum steps
        if (
            "maximum step" in fail_reason_lower
            or "max_steps" in fail_reason_lower
            or "reached maximum" in fail_reason_lower
        ):
            return "false"
        # Model output FAIL
        if "model determined" in fail_reason_lower or "refused to continue" in fail_reason_lower:
            return "false"

    # External errors (API/env) should be null even if task_result == "failure"
    if failure_type in ("api_error", "env_error", "external_error"):
        return None

    # Model faults (parse/action errors) should be false
    if failure_type in ("parse_error", "action_error", "model_error"):
        return "false"

    # If no failure_type and no fail_reason patterns matched, keep as null for potential retry
    if fail_reason:
        pass

    # If task_result == "failure" but no specific failure_type or pattern matched,
    # it might be an unknown error, keep as null for potential retry
    if task_result == "failure":
        return None

    return None


def classify_failure_outcome(
    csv_status: Optional[str], failure_type: Optional[str]
) -> Tuple[str, str]:
    """Classify a failed run into model/external bucket for benchmark stats."""
    if csv_status == "false":
        if failure_type == "parse_error":
            return "failed_model", "Parse error (model)"
        if failure_type == "action_error":
            return "failed_model", "Invalid action (model)"
        if failure_type == "model_error":
            return "failed_model", "Task failure (model)"
        return "failed_model", "Evaluate failed after run"

    if failure_type == "api_error":
        return "failed_external", "API error (external)"
    if failure_type == "env_error":
        return "failed_external", "Env crash (external)"
    return "failed_external", "External error (null)"


def write_missing_result_diagnostic(task_output_dir: str, task_id: str, return_code: Optional[int],
                                    stdout_text: str, stderr_text: str, error_text: str = None):
    """Write a clear diagnostic file when task folder exists but no result JSON was produced."""
    try:
        diag_path = Path(task_output_dir) / "run_error.txt"
        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if return_code is not None:
                f.write(f"Return code: {return_code}\n")
            if error_text:
                f.write(f"Error: {error_text}\n")
            f.write(f"{'=' * 80}\n")
            f.write("No result JSON found in task output directory.\n")
            f.write("This usually means environment/process failed before result serialization.\n\n")

            if stdout_text:
                f.write("=== STDOUT (tail 200 lines) ===\n")
                stdout_lines = stdout_text.splitlines()
                f.write("\n".join(stdout_lines[-200:]))
                f.write("\n\n")

            if stderr_text:
                f.write("=== STDERR (tail 200 lines) ===\n")
                stderr_lines = stderr_text.splitlines()
                f.write("\n".join(stderr_lines[-200:]))
                f.write("\n")
    except Exception as e:
        print(f"  ⚠️  Failed to write run_error.txt for {task_id}: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Read task IDs from CSV and run benchmark (parallel or sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --workers 4 --config experiments/configs/virtualhome/config_close_gpt-5.yaml
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --workers 4 --config experiments/configs/virtualhome/config_close_gpt-5.yaml --headless
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --workers 4 --config experiments/configs/virtualhome/config_close_gpt-5.yaml --max-steps 50
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --sequential --config experiments/configs/virtualhome/config_close_gpt-5.yaml --headless
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --sequential --config experiments/configs/virtualhome/config_close_gpt-5.yaml --skip-completed
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome.csv" --workers 1 --config experiments/configs/virtualhome/config_close_gpt-5.yaml --task virtualhome00000
  #                 （CSV   config       ）:
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome-Gemini-2.5-pro.csv" --config experiments/configs/virtualhome/config_close_Gemini-2.5-pro.yaml --save-name Gemini-2.5-pro --workers 4
  python -m scripts.virtualhome.run_benchmark --csv "experiments/csv/virtualhome/Spatial-Annotation-virtualhome-gpt-5.csv" --config experiments/configs/virtualhome/config_close_gpt-5.yaml --save-name gpt-5 --workers 4
"""
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with task IDs",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/virtualhome/config_close_gpt-5.yaml",
        help="Config file path (default: experiments/configs/virtualhome/config_close_gpt-5.yaml)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional; for benchmark runs this is NOT passed to run_task",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output root dir (default: outputs)",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode (CloudRendering, no X11)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tasks sequentially (one by one). Use when API has concurrency limits",
    )

    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip already completed tasks (from latest benchmark dir)",
    )

    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default="outputs_completed",
        help="Output dir for completed tasks (default: outputs_completed)",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Benchmark run name prefix (e.g. Gemini-2.5-pro)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only this task ID (e.g. virtualhome00000)",
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all tasks in CSV regardless of Completed status",
    )

    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="For virtualhome only: do not auto-launch backend, connect to existing service"
    )

    parser.add_argument(
        "--backend-exe",
        type=str,
        default=DEFAULT_BACKEND_EXE,
        help=f"For virtualhome only: VirtualHome.exe path (default: {DEFAULT_BACKEND_EXE})"
    )

    parser.add_argument(
        "--backend-args",
        type=str,
        default=DEFAULT_BACKEND_ARGS,
        help=f'For virtualhome only: backend launch args (default: "{DEFAULT_BACKEND_ARGS}")'
    )

    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=f"For virtualhome only: wait time for backend port ready in seconds (default: {DEFAULT_STARTUP_TIMEOUT})"
    )

    parser.add_argument(
        "--keep-backend",
        action="store_true",
        help="For virtualhome only: keep backend process alive after benchmark exits"
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
    
    env_type_from_config = infer_env_type_from_config(str(config_path))

    if env_type_from_config == "virtualhome" and not args.sequential:
        print("⚠️  VirtualHome benchmark is forced to sequential mode for backend stability")
        args.sequential = True
        args.workers = 1

    backend_proc = None
    if env_type_from_config == "virtualhome":
        vh_host, vh_port = get_virtualhome_endpoint(config_path)
        if args.no_launch:
            print(f"ℹ️  VirtualHome --no-launch: using existing backend at {vh_host}:{vh_port}")
        elif is_port_open(vh_host, vh_port):
            print(f"ℹ️  VirtualHome backend already reachable at {vh_host}:{vh_port}, reusing existing service")
        else:
            print(f"🚀 Launching VirtualHome backend: {args.backend_exe}")
            try:
                backend_proc = launch_backend(
                    args.backend_exe,
                    args.backend_args,
                    port=vh_port,
                )
            except FileNotFoundError as exc:
                print(f"❌ {exc}")
                sys.exit(1)
            print(f"⏳ Waiting for VirtualHome backend: {vh_host}:{vh_port} (timeout {args.startup_timeout}s)")
            if not wait_for_port(vh_host, vh_port, args.startup_timeout):
                print(f"❌ VirtualHome backend {vh_host}:{vh_port} not ready within {args.startup_timeout}s")
                terminate_backend(backend_proc)
                sys.exit(1)
            print(f"✓ VirtualHome backend ready: {vh_host}:{vh_port}")

    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    elif args.all_tasks:
        print(f"📋 Reading all task IDs from CSV: {csv_path}")
        task_ids = read_task_ids_from_csv(str(csv_path), only_null=False)
        if not task_ids:
            print("❌ No task IDs found in CSV")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks in CSV")
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
        completed_tasks = find_completed_tasks(args.output_dir, args.save_name)
        
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

    unique_task_ids, duplicate_task_ids = deduplicate_task_ids(task_ids)
    if duplicate_task_ids:
        print(f"⚠️  Found {len(duplicate_task_ids)} duplicated task IDs in input; duplicates will be skipped")
        if len(duplicate_task_ids) <= 10:
            print(f"  Duplicates: {duplicate_task_ids}")
        else:
            print(f"  Duplicates (first 10): {duplicate_task_ids[:10]} ...")
    task_ids = unique_task_ids
    
    if not task_ids:
        print("❌ No tasks to run")
        sys.exit(0)
    
    temp_config_file = None
    try:
        if not args.sequential:
            raise _SkipSequential()
        if args.sequential:
            print(f"\n{'=' * 80}")
            print(f"🚀 Starting sequential benchmark")
            print(f"{'=' * 80}")
            print(f"Total tasks: {len(task_ids)}")
            print(f"Mode: sequential (one by one)")
            print(f"Config: {config_path}")
            if args.headless:
                print(f"Headless: enabled (CloudRendering)")
            print(f"{'=' * 80}\n")
            
            actual_config_path = config_path
        
            if args.headless:
                print("🖥️  Headless mode (CloudRendering)")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                if "env" not in config_data:
                    config_data["env"] = {}
                config_data["env"]["platform"] = "CloudRendering"
                temp_config_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.yaml',
                    delete=False,
                    encoding='utf-8'
                )
                yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
                temp_config_file.close()
                actual_config_path = Path(temp_config_file.name)
                print(f"✓ Temp config created: {actual_config_path}\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = args.save_name if args.save_name else "benchmark_sequential"
            benchmark_output_dir = os.path.join(args.output_dir, f"{prefix}_{timestamp}")
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
            task_metadata = load_task_metadata(task_id)
            token_stats = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            actual_actions_info = {"actual_action_count": None, "actual_action_text": None}
            failure_reason_detail = None
            task_result_info = {
                "has_result_json": False,
                "task_result": None,
                "failure_type": None,
                "fail_reason": None,
            }
            stdout_text = ""
            stderr_text = ""
            
            task_output_dir = os.path.join(benchmark_output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                "-m",
                "scripts.virtualhome.work.run_task",
                "--config", str(actual_config_path),
                "--tasks", task_id,
                "--output-dir", task_output_dir
            ]

            normalized_task_id = normalize_task_id(task_id)
            with INFLIGHT_TASKS_LOCK:
                if normalized_task_id in INFLIGHT_TASKS:
                    task_status = "failed_external"
                    failed_external += 1
                    failure_reason_detail = "Task already running (dedupe protection)"
                    if HAS_TQDM:
                        tqdm.write(f"  ⚠️  Task {task_id} skipped: already running")
                    else:
                        print(f"  ⚠️  Task {task_id} skipped: already running")
                    task_duration = time.time() - task_start_time
                    task_records.append({
                        'task_id': task_id,
                        'status': task_status,
                        'attempts': 1,
                        'duration': task_duration,
                        'success': task_success,
                        'log_path': None,
                        'golden_actions_count': task_metadata.get('golden_action_count'),
                        'actual_actions_count': actual_actions_info.get('actual_action_count'),
                        'golden_action': task_metadata.get('golden_action_text'),
                        'actual_actions': actual_actions_info.get('actual_action_text'),
                        'instruction': task_metadata.get('instruction'),
                        'prompt_tokens': token_stats['prompt_tokens'],
                        'completion_tokens': token_stats['completion_tokens'],
                        'total_tokens': token_stats['total_tokens'],
                        'failure_reason': failure_reason_detail,
                        'task_result': task_result_info['task_result'],
                    })
                    continue
                INFLIGHT_TASKS.add(normalized_task_id)

            try:
                execution_start_time = time.time()
                
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_child_process_env(),
                    timeout=None
                )
                execution_duration = time.time() - execution_start_time
                stdout_text = result.stdout or ""
                stderr_text = result.stderr or ""
                
                output_lines = []
                if stdout_text:
                    output_lines.extend(stdout_text.splitlines())
                    task_log_content.append(f"=== STDOUT ===\n")
                    task_log_content.append("\n".join(stdout_text.splitlines()))
                    task_log_content.append("\n")
                if stderr_text:
                    output_lines.extend(stderr_text.splitlines())
                    task_log_content.append(f"=== STDERR ===\n")
                    task_log_content.append("\n".join(stderr_text.splitlines()))
                    task_log_content.append("\n")
                
                task_log_content.append(f"=== Run info ===\n")
                task_log_content.append(f"Exit code: {result.returncode}\n")
                task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
                task_log_content.append(f"{'=' * 80}\n\n")
                
                task_log = "\n".join(task_log_content)
                token_stats_text = extract_token_stats_from_text(task_log)

                result_json = find_result_json(task_output_dir)
                task_result_info = read_result_status_info(result_json)
                token_stats_json = extract_token_stats_from_result_json(result_json)
                token_stats = merge_token_stats(token_stats_json, token_stats_text)
                actual_actions_info = extract_actual_actions(result_json)
                if not failure_reason_detail:
                    failure_reason_detail = task_result_info.get("fail_reason")

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
                    if task_result_info.get('task_result') == 'failure':
                        csv_status = 'false'
                    else:
                        csv_status = 'true'
                    csv_extra_fields = build_csv_extra_fields(
                        task_metadata, actual_actions_info, token_stats, failure_reason_detail
                    )
                    if update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    ):
                        if HAS_TQDM:
                            tqdm.write(f"  ✓ CSV updated: {task_id} -> {csv_status} (+extra fields)")
                        else:
                            print(f"  ✓ CSV updated: {task_id} -> {csv_status} (+extra fields)")
                else:
                    failure_reason = determine_failure_reason(result_json_path=result_json)
                    csv_status = decide_csv_status_from_result(task_result_info, failure_reason)
                    failure_reason_detail = task_result_info.get("fail_reason")
                    if result_json:
                        task_status, status_msg = classify_failure_outcome(
                            csv_status, failure_reason
                        )
                        if task_status == "failed_model":
                            failed_model += 1
                        else:
                            failed_external += 1
                        
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        save_failed_log(task_id, task_output_dir, failed_logs_dir, 1, output_lines)
                        if HAS_TQDM:
                            tqdm.write(f"  ⚠️  Task {task_id} {status_msg}")
                        else:
                            print(f"  ⚠️  Task {task_id} {status_msg}")

                        if csv_status == 'false':
                            if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                                copied_to_completed += 1
                        csv_extra_fields = build_csv_extra_fields(
                            task_metadata, actual_actions_info, token_stats, failure_reason_detail
                        )
                        if update_csv_task_record(
                            csv_path,
                            task_id,
                            status=csv_status,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        ):
                            if HAS_TQDM:
                                tqdm.write(f"  ✓ CSV updated: {task_id} -> {csv_status} (+extra fields)")
                            else:
                                print(f"  ✓ CSV updated: {task_id} -> {csv_status} (+extra fields)")
                    else:
                        task_status = "failed_external"
                        failed_external += 1
                        status_msg = "No JSON produced (external)"
                        failure_reason_detail = "No result JSON produced"
                        csv_extra_fields = build_csv_extra_fields(
                            task_metadata, actual_actions_info, token_stats, failure_reason_detail
                        )
                        
                        save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                        save_failed_log(task_id, task_output_dir, failed_logs_dir, 1, output_lines)
                        write_missing_result_diagnostic(
                            task_output_dir,
                            task_id,
                            result.returncode,
                            stdout_text,
                            stderr_text,
                            error_text=failure_reason_detail,
                        )
                        update_csv_task_record(
                            csv_path,
                            task_id,
                            status=None,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                        
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
                task_result_info = read_result_status_info(result_json)
                token_stats_text = extract_token_stats_from_text(task_log)
                actual_actions_info = extract_actual_actions(result_json)
                token_stats_json = extract_token_stats_from_result_json(result_json)
                token_stats = merge_token_stats(token_stats_json, token_stats_text)
                failure_reason_detail = task_result_info.get("fail_reason") or error_msg
                csv_status = decide_csv_status_from_result(task_result_info, failure_reason)
                csv_extra_fields = build_csv_extra_fields(
                    task_metadata, actual_actions_info, token_stats, failure_reason_detail
                )
                
                if failure_reason in ("parse_error", "action_error"):
                    failed_model += 1
                    task_status = "failed_model"
                    if csv_status == 'false' and copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                        copied_to_completed += 1
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                else:
                    failed_external += 1
                    task_status = "failed_external"
                    if csv_status == 'false' and copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                        copied_to_completed += 1
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                    if not result_json:
                        write_missing_result_diagnostic(
                            task_output_dir,
                            task_id,
                            None,
                            "",
                            "",
                            error_text=error_msg,
                        )
                    failure_reason_detail = error_msg
            finally:
                with INFLIGHT_TASKS_LOCK:
                    INFLIGHT_TASKS.discard(normalized_task_id)
            
            task_duration = time.time() - task_start_time
            task_records.append({
                'task_id': task_id,
                'status': task_status,
                'attempts': 1,
                'duration': task_duration,
                'success': task_success,
                'log_path': None,
                'golden_actions_count': task_metadata.get('golden_action_count'),
                'actual_actions_count': actual_actions_info.get('actual_action_count'),
                'golden_action': task_metadata.get('golden_action_text'),
                'actual_actions': actual_actions_info.get('actual_action_text'),
                'instruction': task_metadata.get('instruction'),
                'prompt_tokens': token_stats['prompt_tokens'],
                'completion_tokens': token_stats['completion_tokens'],
                'total_tokens': token_stats['total_tokens'],
                'failure_reason': failure_reason_detail,
                'task_result': task_result_info['task_result'],
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
                    golden_action_count = record.get('golden_actions_count')
                    actual_action_count = record.get('actual_actions_count')
                    total_tokens = record.get('total_tokens')
                    golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
                    actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
                    token_str = str(total_tokens) if total_tokens is not None else "N/A"
                    f.write(
                        f"  ✅ {record['task_id']}: {record['duration']:.2f}s, {record['attempts']} attempt(s), "
                        f"golden_actions: {golden_action_str}, actual_actions: {actual_action_str}, tokens: {token_str}\n"
                    )
                f.write(f"\n")
        
            if failed_tasks:
                f.write(f"Failed ({len(failed_tasks)}):\n")
                f.write(f"{'-' * 80}\n")
                for record in failed_tasks:
                    golden_action_count = record.get('golden_actions_count')
                    actual_action_count = record.get('actual_actions_count')
                    total_tokens = record.get('total_tokens')
                    golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
                    actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
                    token_str = str(total_tokens) if total_tokens is not None else "N/A"
                    fail_reason = record.get('failure_reason') or "N/A"
                    f.write(
                        f"  ❌ {record['task_id']}: {record['duration']:.2f}s, {record['attempts']} attempt(s), "
                        f"golden_actions: {golden_action_str}, actual_actions: {actual_action_str}, tokens: {token_str}, reason: {fail_reason}\n"
                    )
                f.write(f"\n")
        
            f.write(f"{'=' * 80}\n")
            f.write(f"All tasks\n")
            f.write(f"{'=' * 80}\n")
            for i, record in enumerate(task_records, 1):
                status_icon = "✅" if record['success'] else "❌"
                golden_action_count = record.get('golden_actions_count')
                actual_action_count = record.get('actual_actions_count')
                total_tokens = record.get('total_tokens')
                golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
                actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
                token_str = str(total_tokens) if total_tokens is not None else "N/A"
                f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                       f"status: {record['status']:10s} | "
                       f"attempts: {record['attempts']:2d} | "
                       f"duration: {record['duration']:8.2f}s | "
                      f"golden_actions: {golden_action_str:>4s} | "
                      f"actual_actions: {actual_action_str:>4s} | "
                       f"tokens: {token_str:>8s}")
                if record['log_path']:
                    f.write(f" | log: {record['log_path']}")
                if record.get('failure_reason') and not record['success']:
                    f.write(f" | reason: {record['failure_reason']}")
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
    except _SkipSequential:
        pass
    finally:
        if temp_config_file and os.path.exists(temp_config_file.name):
            try:
                os.unlink(temp_config_file.name)
                print("✓ Temp config removed")
            except Exception as e:
                print(f"⚠️ Failed to remove temp config: {e}")
        if backend_proc and not args.keep_backend:
            print("🛑 Stopping VirtualHome backend...")
            terminate_backend(backend_proc)
            print("✓ VirtualHome backend stopped")
        elif backend_proc and args.keep_backend:
            print("ℹ️  --keep-backend enabled: keeping VirtualHome backend process alive")
    
    actual_config_path = config_path
    temp_config_file = None

    if args.headless:
        print("🖥️  Headless mode (CloudRendering)")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if "env" not in config_data:
            config_data["env"] = {}
        config_data["env"]["platform"] = "CloudRendering"
        temp_config_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False,
            encoding='utf-8'
        )
        yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
        temp_config_file.close()
        
        actual_config_path = Path(temp_config_file.name)
        print(f"✓ Temp config created: {actual_config_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.save_name if args.save_name else "benchmark"
    benchmark_output_dir = os.path.join(args.output_dir, f"{prefix}_{timestamp}")
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
        print(f"Headless: enabled (CloudRendering)")
    print(f"{'=' * 80}\n")
    
    def execute_task(task_id: str) -> dict:
        """Run one task; on success copy to outputs_completed and update CSV."""
        task_start_time = time.time()
        task_log_content = []
        task_status = "failed"
        task_success = False
        task_metadata = load_task_metadata(task_id)
        token_stats = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        actual_actions_info = {"actual_action_count": None, "actual_action_text": None}
        failure_reason_detail = None
        task_result_info = {
            "has_result_json": False,
            "task_result": None,
            "failure_type": None,
            "fail_reason": None,
        }
        task_output_dir = os.path.join(benchmark_output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)

        normalized_task_id = normalize_task_id(task_id)
        with INFLIGHT_TASKS_LOCK:
            if normalized_task_id in INFLIGHT_TASKS:
                return {
                    'task_id': task_id,
                    'status': 'failed_external',
                    'attempts': 1,
                    'duration': time.time() - task_start_time,
                    'success': False,
                    'log_path': None,
                    'golden_actions_count': task_metadata.get('golden_action_count'),
                    'actual_actions_count': actual_actions_info.get('actual_action_count'),
                    'golden_action': task_metadata.get('golden_action_text'),
                    'actual_actions': actual_actions_info.get('actual_action_text'),
                    'instruction': task_metadata.get('instruction'),
                    'prompt_tokens': token_stats['prompt_tokens'],
                    'completion_tokens': token_stats['completion_tokens'],
                    'total_tokens': token_stats['total_tokens'],
                    'failure_reason': 'Task already running (dedupe protection)',
                    'task_result': None,
                }
            INFLIGHT_TASKS.add(normalized_task_id)
        
        cmd = [
            sys.executable,
            "-m",
                "scripts.virtualhome.work.run_task",
            "--config", str(actual_config_path),
            "--tasks", task_id,
            "--output-dir", task_output_dir
        ]
        try:
            execution_start_time = time.time()
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_child_process_env(),
                timeout=None
            )
            execution_duration = time.time() - execution_start_time
            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""
            
            if stdout_text:
                task_log_content.append(f"=== STDOUT ===\n")
                task_log_content.append(stdout_text)
                task_log_content.append("\n")
            if stderr_text:
                task_log_content.append(f"=== STDERR ===\n")
                task_log_content.append(stderr_text)
                task_log_content.append("\n")
            
            task_log_content.append(f"=== Run info ===\n")
            task_log_content.append(f"Exit code: {result.returncode}\n")
            task_log_content.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_content.append(f"{'=' * 80}\n\n")
            
            task_log = "".join(task_log_content)
            token_stats_text = extract_token_stats_from_text(task_log)
            result_json = find_result_json(task_output_dir)
            task_result_info = read_result_status_info(result_json)
            token_stats_json = extract_token_stats_from_result_json(result_json)
            token_stats = merge_token_stats(token_stats_json, token_stats_text)
            actual_actions_info = extract_actual_actions(result_json)
            if not failure_reason_detail:
                failure_reason_detail = task_result_info.get("fail_reason")

            if check_task_success(task_id, task_output_dir, str(actual_config_path), args.headless):
                task_success = True
                task_status = "success"
                save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                if copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                    if task_result_info.get('task_result') == 'failure':
                        csv_status = 'false'
                    else:
                        csv_status = 'true'
                    csv_extra_fields = build_csv_extra_fields(
                        task_metadata, actual_actions_info, token_stats, failure_reason_detail
                    )
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                    print(f"  ✅ {task_id} success (evaluate passed) -> copied and CSV updated({csv_status}, +extra)")
            else:
                failure_reason = determine_failure_reason(result_json_path=result_json)
                csv_status = decide_csv_status_from_result(task_result_info, failure_reason)
                failure_reason_detail = task_result_info.get("fail_reason")
                csv_extra_fields = build_csv_extra_fields(
                    task_metadata, actual_actions_info, token_stats, failure_reason_detail
                )
                if result_json:
                    task_status, status_msg = classify_failure_outcome(
                        csv_status, failure_reason
                    )
                    save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                    if task_status == "failed_model":
                        if csv_status == 'false' and copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                            update_csv_task_record(
                                csv_path,
                                task_id,
                                status=csv_status,
                                extra_fields=csv_extra_fields,
                                lock=csv_lock,
                            )
                            print(f"  ❌ {task_id} {status_msg} -> {csv_status}")
                        else:
                            update_csv_task_record(
                                csv_path,
                                task_id,
                                status=csv_status,
                                extra_fields=csv_extra_fields,
                                lock=csv_lock,
                            )
                    else:
                        update_csv_task_record(
                            csv_path,
                            task_id,
                            status=None,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                        print(f"  ⚠️  {task_id} {status_msg}")
                else:
                    task_status = "failed_external"
                    save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                    print(f"  ⚠️  {task_id} No JSON -> null")
                    failure_reason_detail = "No result JSON produced"
                    csv_extra_fields = build_csv_extra_fields(
                        task_metadata, actual_actions_info, token_stats, failure_reason_detail
                    )
                    write_missing_result_diagnostic(
                        task_output_dir,
                        task_id,
                        result.returncode,
                        stdout_text,
                        stderr_text,
                        error_text=failure_reason_detail,
                    )
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=None,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
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
            task_result_info = read_result_status_info(result_json)
            token_stats_text = extract_token_stats_from_text(task_log)
            actual_actions_info = extract_actual_actions(result_json)
            token_stats_json = extract_token_stats_from_result_json(result_json)
            token_stats = merge_token_stats(token_stats_json, token_stats_text)
            failure_reason_detail = task_result_info.get("fail_reason") or str(e)
            csv_status = decide_csv_status_from_result(task_result_info, failure_reason)
            csv_extra_fields = build_csv_extra_fields(
                task_metadata, actual_actions_info, token_stats, failure_reason_detail
            )
            task_status, _status_msg = classify_failure_outcome(csv_status, failure_reason)
            if task_status == "failed_model":
                if csv_status == 'false' and copy_to_outputs_completed(task_id, task_output_dir, str(outputs_completed_path)):
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                    print(f"  ❌ {task_id} Model failure -> {csv_status}")
                else:
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status=csv_status,
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
            else:
                task_status = "failed_external"
                print(f"  ⚠️  {task_id} External -> null")
                update_csv_task_record(
                    csv_path,
                    task_id,
                    status=None,
                    extra_fields=csv_extra_fields,
                    lock=csv_lock,
                )
                if not result_json:
                    write_missing_result_diagnostic(
                        task_output_dir,
                        task_id,
                        None,
                        "",
                        "",
                        error_text=str(e),
                    )
        finally:
            with INFLIGHT_TASKS_LOCK:
                INFLIGHT_TASKS.discard(normalized_task_id)

        task_duration = time.time() - task_start_time
        return {
            'task_id': task_id,
            'status': task_status,
            'attempts': 1,
            'duration': task_duration,
            'success': task_success,
            'log_path': None,
            'golden_actions_count': task_metadata.get('golden_action_count'),
            'actual_actions_count': actual_actions_info.get('actual_action_count'),
            'golden_action': task_metadata.get('golden_action_text'),
            'actual_actions': actual_actions_info.get('actual_action_text'),
            'instruction': task_metadata.get('instruction'),
            'prompt_tokens': token_stats['prompt_tokens'],
            'completion_tokens': token_stats['completion_tokens'],
            'total_tokens': token_stats['total_tokens'],
            'failure_reason': failure_reason_detail,
            'task_result': task_result_info['task_result'],
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
                golden_action_count = record.get('golden_actions_count')
                actual_action_count = record.get('actual_actions_count')
                total_tokens = record.get('total_tokens')
                golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
                actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
                token_str = str(total_tokens) if total_tokens is not None else "N/A"
                f.write(
                    f"  ✅ {record['task_id']}: {record['duration']:.2f}s, "
                    f"golden_actions: {golden_action_str}, actual_actions: {actual_action_str}, tokens: {token_str}\n"
                )
            f.write(f"\n")
        
        if failed_tasks:
            f.write(f"Failed ({len(failed_tasks)}):\n")
            f.write(f"{'-' * 80}\n")
            for record in failed_tasks:
                golden_action_count = record.get('golden_actions_count')
                actual_action_count = record.get('actual_actions_count')
                total_tokens = record.get('total_tokens')
                golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
                actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
                token_str = str(total_tokens) if total_tokens is not None else "N/A"
                fail_reason = record.get('failure_reason') or "N/A"
                f.write(
                    f"  ❌ {record['task_id']}: {record['duration']:.2f}s, "
                    f"golden_actions: {golden_action_str}, actual_actions: {actual_action_str}, "
                    f"tokens: {token_str}, reason: {fail_reason}\n"
                )
            f.write(f"\n")
        
        f.write(f"{'=' * 80}\n")
        f.write(f"All tasks\n")
        f.write(f"{'=' * 80}\n")
        for i, record in enumerate(task_records, 1):
            status_icon = "✅" if record['success'] else "❌"
            golden_action_count = record.get('golden_actions_count')
            actual_action_count = record.get('actual_actions_count')
            total_tokens = record.get('total_tokens')
            golden_action_str = str(golden_action_count) if golden_action_count is not None else "N/A"
            actual_action_str = str(actual_action_count) if actual_action_count is not None else "N/A"
            token_str = str(total_tokens) if total_tokens is not None else "N/A"
            f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                   f"status: {record['status']:10s} | "
                   f"duration: {record['duration']:8.2f}s | "
                 f"golden_actions: {golden_action_str:>4s} | "
                 f"actual_actions: {actual_action_str:>4s} | "
                   f"tokens: {token_str:>8s}")
            if record['log_path']:
                f.write(f" | log: {record['log_path']}")
            if record.get('failure_reason') and not record['success']:
                f.write(f" | reason: {record['failure_reason']}")
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

    if backend_proc and not args.keep_backend:
        print("🛑 Stopping VirtualHome backend...")
        terminate_backend(backend_proc)
        print("✓ VirtualHome backend stopped")
    elif backend_proc and args.keep_backend:
        print("ℹ️  --keep-backend enabled: keeping VirtualHome backend process alive")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
