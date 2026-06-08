#!/usr/bin/env python3
"""
 CSV      ID         
"""

import io
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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional, Tuple, Dict, Any

if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name)
        if hasattr(_stream, "buffer"):
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace"),
            )

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm   ，              : pip install tqdm")


def _run_capture_utf8(cmd: list, timeout=None):
    """Run subprocess and decode stdout/stderr as UTF-8.

    On Windows, ``text=True`` without reliable UTF-8 can use GBK and crash
    (UnicodeDecodeError) when child prints emoji or UTF-8 bytes. Capturing
    bytes and decoding here avoids that; also nudges child Python toward UTF-8.
    """
    from types import SimpleNamespace

    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    return SimpleNamespace(
        returncode=proc.returncode,
        stdout=(proc.stdout or b"").decode("utf-8", errors="replace"),
        stderr=(proc.stderr or b"").decode("utf-8", errors="replace"),
    )


def _local_model_health_target(config_path: Path) -> Optional[Tuple[str, Dict[str, str]]]:
    """Return a /models URL for local OpenAI-compatible base_url, else None."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    vlm = (data.get("model") or {}).get("vlm") or {}
    base_url = str(vlm.get("base_url") or "").strip()
    if not base_url:
        return None

    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None

    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    health_path = f"{path}/models" if path else "/models"
    health_url = urllib.parse.urlunparse(
        (parsed.scheme or "http", parsed.netloc, health_path, "", "", "")
    )

    headers = {}
    api_key = str(vlm.get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return health_url, headers


def wait_for_local_model_api(
    config_path: Path,
    timeout_seconds: float,
    interval_seconds: float,
) -> bool:
    """Wait for a local OpenAI-compatible API before burning a task attempt."""
    if timeout_seconds <= 0:
        return True

    target = _local_model_health_target(config_path)
    if target is None:
        return True

    health_url, headers = target
    deadline = time.time() + timeout_seconds
    next_log_at = 0.0
    last_error = "unknown error"

    while True:
        try:
            req = urllib.request.Request(health_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if int(getattr(resp, "status", 200)) < 500:
                    return True
        except urllib.error.HTTPError as e:
            # Auth/client errors still prove the tunnel and server are reachable.
            if int(e.code) < 500:
                return True
            last_error = f"HTTP {e.code}"
        except Exception as e:
            last_error = str(e)

        remaining = deadline - time.time()
        if remaining <= 0:
            print(
                f"⚠️  Model API unavailable after {timeout_seconds:.0f}s: "
                f"{health_url} ({last_error})"
            )
            return False

        now = time.time()
        if now >= next_log_at:
            print(
                f"⏳ Waiting for model API tunnel: {health_url} "
                f"(last error: {last_error}; {remaining:.0f}s left)"
            )
            next_log_at = now + 30
        time.sleep(min(interval_seconds, max(0.1, remaining)))


def read_task_ids_from_csv(csv_path: str, only_pending: bool = False) -> list:
    """
     CSV       ID 

    Completed   （  Spatial-Annotation     ）：
    - true:    evaluate /    JSON       ，   
    - false:      （DONE/FAIL/max_steps     ACTION     ACTION/     ），   
    - null   :       ，  API/         （graph              -> false）

    Args:
        csv_path: CSV    
        only_pending:   True      Completed   null/    （   true   false）

    Returns:
          ID  
    """
    task_ids = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        completed_col_idx = None
        if header and len(header) > 1:
            for idx, col in enumerate(header):
                if col.strip().lower() in ["completed", "    ", "status"]:
                    completed_col_idx = idx
                    break

        for row in reader:
            if not row or not row[0].strip():
                continue

            task_id = row[0].strip()
            if task_id == "Task ID":
                continue

            if only_pending:
                status = ""
                if completed_col_idx is not None and len(row) > completed_col_idx:
                    status = row[completed_col_idx].strip().lower()
                #   null /       ；true false    
                if status in ("true", "false"):
                    continue
                if status not in ("", "null"):
                    #       ：         ，    
                    pass

            task_ids.append(task_id)

    return task_ids


def find_completed_tasks(output_dir: str) -> set:
    """
            （       benchmark    log.json episode_*.json  ）
    
    Args:
        output_dir:      
        
    Returns:
              ID  
    """
    completed = set()
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return completed
    
    #     benchmark  （    ），      ，     
    benchmark_dirs = sorted(
        [d for d in output_path.glob("benchmark_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    #            
    sequential_dirs = sorted(
        [d for d in output_path.glob("benchmark_sequential_*") if d.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    
    #       ，       
    all_dirs = benchmark_dirs + sequential_dirs
    
    for benchmark_dir in all_dirs:
        #        worker  
        for worker_dir in benchmark_dir.glob("worker_*"):
            if worker_dir.is_dir():
                #         
                for task_dir in worker_dir.iterdir():
                    if task_dir.is_dir():
                        task_id = task_dir.name
                        if check_eval_run_finished(str(task_dir)):
                            completed.add(task_id)
                        else:
                            #      （run_*  ），      task_id/run_*/task_id/
                            for run_dir in task_dir.glob("run_*"):
                                if run_dir.is_dir():
                                    task_subdir = run_dir / task_id
                                    if task_subdir.exists() and check_eval_run_finished(str(task_subdir)):
                                        completed.add(task_id)
                                        break
                                    if check_eval_run_finished(str(run_dir)):
                                        completed.add(task_id)
                                        break
                        
                        #   ：    png    json  ，          ，         
        
        #            （   benchmark   ）
        for task_dir in benchmark_dir.iterdir():
            if task_dir.is_dir() and not task_dir.name.startswith("worker_"):
                task_id = task_dir.name
                #      log.json episode_*.json
                #   ：     ，          task_id/run_*/task_id/
                #       （run_*  ）
                for run_dir in task_dir.glob("run_*"):
                    if run_dir.is_dir():
                        task_subdir = run_dir / task_id
                        if task_subdir.exists() and check_eval_run_finished(str(task_subdir)):
                            completed.add(task_id)
                            break
                        if check_eval_run_finished(str(run_dir)):
                            completed.add(task_id)
                            break
                
                if check_eval_run_finished(str(task_dir)):
                    completed.add(task_id)
    
    return completed


def model_task_succeeded(result_info: dict) -> bool:
    """          （     ）    decide_csv_status   task_success    """
    if not result_info.get("has_json"):
        return False
    if result_info.get("success") is True:
        return True
    tr = str(result_info.get("task_result") or "").strip().lower()
    if tr in {"success", "succeeded", "true"}:
        return True
    return False


def check_eval_run_finished(task_output_dir: str) -> bool:
    """
       python -m scripts.carla.work.run_task               JSON（      ） 
           (success: false)         ，             
    """
    result_json = find_result_json(task_output_dir)
    if not result_json or not result_json.exists():
        return False
    info = read_result_status_info(result_json)
    return bool(info.get("has_json"))


def check_task_success(task_output_dir: str) -> bool:
    """
               
    
    Args:
        task_output_dir:       
        
    Returns:
            JSON          True，    False
    """
    task_path = Path(task_output_dir)

    def read_success_from_json(json_path: Path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        success = data.get("success")
        if isinstance(success, bool):
            return success

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            meta_success = metadata.get("success")
            if isinstance(meta_success, bool):
                return meta_success

            task_result = str(metadata.get("task_result", "")).strip().lower()
            if task_result in {"success", "succeeded", "true"}:
                return True
            if task_result in {"failure", "failed", "false"}:
                return False

        return None

    def candidate_json_files(base_dir: Path):
        candidates = []

        direct_log = base_dir / "log.json"
        if direct_log.exists():
            candidates.append(direct_log)
        candidates.extend(sorted(base_dir.glob("episode_*.json")))

        task_name_with_retry = base_dir.name
        task_id_match = re.match(r"^(ai2thor\d+|carla\d+)(?:_retry_\d+)?$", task_name_with_retry)
        task_id = task_id_match.group(1) if task_id_match else None

        for run_dir in sorted(base_dir.glob("run_*")):
            if not run_dir.is_dir():
                continue

            run_log = run_dir / "log.json"
            if run_log.exists():
                candidates.append(run_log)
            candidates.extend(sorted(run_dir.glob("episode_*.json")))

            if task_id:
                task_subdir = run_dir / task_id
                if task_subdir.exists():
                    sub_log = task_subdir / "log.json"
                    if sub_log.exists():
                        candidates.append(sub_log)
                    candidates.extend(sorted(task_subdir.glob("episode_*.json")))

            for subdir in sorted(run_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                sub_log = subdir / "log.json"
                if sub_log.exists():
                    candidates.append(sub_log)
                candidates.extend(sorted(subdir.glob("episode_*.json")))

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            candidate_str = str(candidate.resolve())
            if candidate_str not in seen:
                seen.add(candidate_str)
                unique_candidates.append(candidate)
        return unique_candidates

    explicit_failure_found = False
    for json_file in candidate_json_files(task_path):
        success = read_success_from_json(json_file)
        if success is True:
            return True
        if success is False:
            explicit_failure_found = True

    if explicit_failure_found:
        return False

    return False


def find_result_json(task_output_dir: str) -> Optional[Path]:
    """          JSON（  log.json，  episode_*.json）"""
    task_path = Path(task_output_dir)
    if not task_path.exists():
        return None

    direct_log = task_path / "log.json"
    if direct_log.exists():
        return direct_log

    direct_episodes = sorted(task_path.glob("episode_*.json"))
    if direct_episodes:
        return direct_episodes[-1]

    task_name_with_retry = task_path.name
    task_id_match = re.match(r"^(ai2thor\d+|carla\d+)(?:_retry_\d+)?$", task_name_with_retry)
    task_id = task_id_match.group(1) if task_id_match else None

    for run_dir in sorted(task_path.glob("run_*")):
        if not run_dir.is_dir():
            continue

        run_log = run_dir / "log.json"
        if run_log.exists():
            return run_log

        run_episodes = sorted(run_dir.glob("episode_*.json"))
        if run_episodes:
            return run_episodes[-1]

        if task_id:
            task_subdir = run_dir / task_id
            if task_subdir.exists():
                sub_log = task_subdir / "log.json"
                if sub_log.exists():
                    return sub_log
                sub_episodes = sorted(task_subdir.glob("episode_*.json"))
                if sub_episodes:
                    return sub_episodes[-1]

        for subdir in sorted(run_dir.iterdir()):
            if not subdir.is_dir():
                continue
            sub_log = subdir / "log.json"
            if sub_log.exists():
                return sub_log
            sub_episodes = sorted(subdir.glob("episode_*.json"))
            if sub_episodes:
                return sub_episodes[-1]

    return None


def read_result_status_info(result_json_path: Optional[Path]) -> dict:
    """    JSON      """
    info = {
        "has_json": False,
        "success": None,
        "task_result": None,
        "failure_type": None,
        "fail_reason": None,
        "action_sequence": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    if not result_json_path or not result_json_path.exists():
        return info

    info["has_json"] = True
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return info

    if not isinstance(data, dict):
        return info

    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    info["success"] = data.get("success")
    info["task_result"] = metadata.get("task_result")
    info["failure_type"] = data.get("failure_type") or metadata.get("failure_type")
    info["fail_reason"] = data.get("fail_reason") or metadata.get("fail_reason")
    info["action_sequence"] = data.get("action_sequence")

    #          token
    token_sources = [
        data.get("token_usage"),
        metadata.get("token_usage"),
        data.get("usage"),
        metadata.get("usage"),
        data,
        metadata,
    ]
    for src in token_sources:
        if not isinstance(src, dict):
            continue
        if info["prompt_tokens"] is None and isinstance(
            src.get("prompt_tokens"), (int, float)
        ):
            info["prompt_tokens"] = int(src.get("prompt_tokens"))
        if info["completion_tokens"] is None and isinstance(
            src.get("completion_tokens"), (int, float)
        ):
            info["completion_tokens"] = int(src.get("completion_tokens"))
        if info["total_tokens"] is None and isinstance(
            src.get("total_tokens"), (int, float)
        ):
            info["total_tokens"] = int(src.get("total_tokens"))

    return info


def read_task_metadata(task_id: str) -> dict:
    """      ，   instruction   golden_actions"""
    result = {
        "instruction": None,
        "golden_action": None,
        "golden_actions_count": None,
    }
    task_json = Path(os.environ.get("CARLA_TASKS_ROOT", "data/carla/tasks")) / task_id / "task.json"
    if not task_json.exists():
        return result

    try:
        with open(task_json, "r", encoding="utf-8") as f:
            task_data = json.load(f)
    except Exception:
        return result

    if not isinstance(task_data, dict):
        return result

    result["instruction"] = task_data.get("instruction")

    golden = task_data.get("golden_actions")
    if isinstance(golden, dict):
        actions = golden.get("actions")
        if isinstance(actions, list):
            result["golden_action"] = "->".join(str(a) for a in actions)
            result["golden_actions_count"] = len(actions)
            return result

    if isinstance(golden, list):
        result["golden_action"] = "->".join(str(a) for a in golden)
        result["golden_actions_count"] = len(golden)
    elif isinstance(golden, str):
        result["golden_action"] = golden
        result["golden_actions_count"] = len([x for x in golden.split("->") if x.strip()])

    return result


def parse_action_sequence(action_sequence: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """  action_sequence       """
    if action_sequence is None:
        return None, None

    action_text = str(action_sequence).strip()
    if not action_text or action_text == "(no action records)":
        return action_text, 0

    actions = [a.strip() for a in action_text.split("->") if a.strip()]
    return action_text, len(actions)


def _fail_reason_is_action_format_exhausted(fail_reason: str) -> bool:
    """
    graph think_node              fail_reason，    （         ACTION） 
     : Step 5 failed after 3 retries: Missing <ACTION> tag or empty ACTION content
    """
    s = fail_reason.lower()
    if "missing <action> tag or empty action content" in s:
        return True
    if "action tag content is empty" in s:
        return True
    if "action parsing error:" in s:
        return True
    return False


def _is_csv_model_failure_reason(fail_reason: Optional[str]) -> bool:
    """
    CSV   Completed=false：          （  core/agent/graph.py   fail_reason   ） 
    - DONE     / FAIL / max_steps
    -    3 step     ACTION（    ）
    -     think         ACTION   ACTION ACTION      （Step N failed after ...）
    - graph     （agent        ；       success=false，       ）
       （API    ）  null 
    """
    fr = (fail_reason or "").strip()
    if not fr:
        return False
    fr_lower = fr.lower()
    # python -m scripts.carla.work.run_task GraphRecursionError：    "Reached recursion limit (100 steps)"，     "graph recursion limit"
    if "recursion limit" in fr_lower:
        return True
    if fr == "Model claimed DONE but success conditions not met":
        return True
    if fr == "Model determined task cannot be completed or refused to continue":
        return True
    if fr.startswith("Reached maximum step limit ("):
        return True
    if fr == "Model failed to provide a valid <ACTION> in 3 consecutive steps":
        return True
    if _fail_reason_is_action_format_exhausted(fr):
        return True
    return False


def decide_csv_status(result_info: dict, task_success: bool) -> str:
    """Completed：true=   evaluate；false=        （     ACTION）；null=   """
    if not result_info.get("has_json"):
        return "null"

    if task_success:
        return "true"

    task_result = str(result_info.get("task_result") or "").strip().lower()
    if task_result in {"success", "succeeded", "true"}:
        return "true"

    success = result_info.get("success")
    if success is True:
        return "true"

    fr = (result_info.get("fail_reason") or "").strip()
    if _is_csv_model_failure_reason(fr):
        return "false"

    return "null"


def should_retry_csv_status(csv_status: str, attempt: int, max_retries: int) -> bool:
    """Retry infra/API outcomes that intentionally remain CSV null."""
    return csv_status == "null" and attempt < max_retries


def update_csv_task_record(
    csv_path: Path,
    task_id: str,
    status: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    lock: Lock = None,
) -> bool:
    """  CSV   ：Completed   +      """
    if lock:
        lock.acquire()

    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        if not rows:
            return False

        header = rows[0]

        def ensure_col(col_name: str) -> int:
            for idx, col in enumerate(header):
                if col.strip().lower() == col_name.strip().lower():
                    return idx
            header.append(col_name)
            new_idx = len(header) - 1
            for row_i in range(1, len(rows)):
                while len(rows[row_i]) <= new_idx:
                    rows[row_i].append("")
            return new_idx

        completed_col_idx = None
        for idx, col in enumerate(header):
            if col.strip().lower() in ["completed", "    ", "status"]:
                completed_col_idx = idx
                break
        if completed_col_idx is None:
            completed_col_idx = ensure_col("Completed")

        extra_idx_map = {}
        if extra_fields:
            for field_name in extra_fields.keys():
                extra_idx_map[field_name] = ensure_col(field_name)

        normalized_task_id = normalize_task_id(task_id)
        updated = False
        for i in range(1, len(rows)):
            if not rows[i] or not rows[i][0].strip():
                continue
            csv_task_id = rows[i][0].strip()
            normalized_csv_id = normalize_task_id(csv_task_id)
            if normalized_csv_id != normalized_task_id and csv_task_id != task_id:
                continue

            max_needed = max([completed_col_idx] + list(extra_idx_map.values()) if extra_idx_map else [completed_col_idx])
            while len(rows[i]) <= max_needed:
                rows[i].append("")

            if status is not None:
                rows[i][completed_col_idx] = status
                updated = True
            elif rows[i][completed_col_idx] != "":
                rows[i][completed_col_idx] = ""
                updated = True

            if extra_fields:
                for field_name, value in extra_fields.items():
                    idx = extra_idx_map[field_name]
                    value_str = "" if value is None else str(value)
                    if rows[i][idx] != value_str:
                        rows[i][idx] = value_str
                        updated = True
            break

        if not updated:
            return False

        backup_path = csv_path.with_suffix(".csv.backup")
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return True
    except Exception as e:
        print(f"  ⚠️    CSV     ({task_id}): {e}")
        return False
    finally:
        if lock:
            lock.release()


def copy_to_outputs_true(task_id: str, task_output_dir: str, outputs_true_dir: str) -> bool:
    """
             outputs_true  
    
    Args:
        task_id:   ID（  retry  ，   "ai2thor03023"）
        task_output_dir:       （   retry  ，   "ai2thor03023_retry_2"）
        outputs_true_dir: outputs_true    
        
    Returns:
              True，    False
    """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            print(f"  ⚠️        : {task_output_dir}")
            return False
        
        #       task_id（  retry  ）
        dest_path = Path(outputs_true_dir) / task_id
        
        #        ，   
        if dest_path.exists():
            shutil.rmtree(dest_path)
        
        #       
        shutil.copytree(source_path, dest_path)
        return True
    except Exception as e:
        print(f"  ❌    outputs_true   ({task_id}): {e}")
        return False


def normalize_task_id(task_id: str) -> str:
    """
         ID  （     ，    ）
    
      ：
    - ai2thor_04000 -> ai2thor04000
    - ai2thor04000 -> ai2thor04000
    """
    if '_' in task_id and task_id.startswith('ai2thor_'):
        return task_id.replace('ai2thor_', 'ai2thor', 1)
    return task_id


def update_csv_task_status(csv_path: Path, task_id: str, status: str = 'true', lock: Lock = None):
    """
      CSV        Completed  
    
    Args:
        csv_path: CSV    
        task_id:   ID
        status:       （'true'   'false'）
        lock:    （            ）
    """
    if lock:
        lock.acquire()
    
    try:
        #   CSV  
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        
        if not rows:
            return False
        
        #   Completed    
        header = rows[0]
        completed_col_idx = None
        for idx, col in enumerate(header):
            if col.strip().lower() in ['completed', '    ', 'status']:
                completed_col_idx = idx
                break
        
        #     Completed ，    
        if completed_col_idx is None:
            header.append("Completed")
            completed_col_idx = len(header) - 1
            for i in range(1, len(rows)):
                if len(rows[i]) <= completed_col_idx:
                    rows[i].extend([''] * (completed_col_idx + 1 - len(rows[i])))
        
        #      ID    
        normalized_task_id = normalize_task_id(task_id)
        
        #         
        updated = False
        for i in range(1, len(rows)):
            if not rows[i] or not rows[i][0].strip():
                continue
            
            csv_task_id = rows[i][0].strip()
            normalized_csv_id = normalize_task_id(csv_task_id)
            
            #     ID（      ）
            if normalized_csv_id == normalized_task_id or csv_task_id == task_id:
                #         
                while len(rows[i]) <= completed_col_idx:
                    rows[i].append('')
                
                old_status = rows[i][completed_col_idx].strip().lower() if rows[i][completed_col_idx] else ''
                if old_status != status.lower():
                    rows[i][completed_col_idx] = status
                    updated = True
                    break
        
        #       CSV
        if updated:
            #     （        ）
            backup_path = csv_path.with_suffix('.csv.backup')
            if not backup_path.exists():
                shutil.copy2(csv_path, backup_path)
            
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            return True
        
        return False
    except Exception as e:
        print(f"  ⚠️    CSV   ({task_id}): {e}")
        return False
    finally:
        if lock:
            lock.release()


def save_task_log(task_id: str, log_content: str, logs_dir: Path, attempt: int = None, status: str = "unknown"):
    """
               
    
    Args:
        task_id:   ID
        log_content:     
        logs_dir:     
        attempt:     （  ）
        status:     （success/failed/unknown）
    """
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        if attempt:
            log_filename = f"{task_id}_attempt_{attempt}_{status}.log"
        else:
            log_filename = f"{task_id}_{status}.log"
        
        log_path = logs_dir / log_filename
        with open(log_path, 'w', encoding='utf-8') as f:
            #         
            f.write(f"  ID: {task_id}\n")
            f.write(f"  : {status}\n")
            if attempt:
                f.write(f"    : {attempt}\n")
            f.write(f"  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(log_content)
        
        return log_path
    except Exception as e:
        print(f"  ⚠️           ({task_id}): {e}")
        return None


def save_failed_log(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int, output_lines: list = None):
    """
             
    
    Args:
        task_id:   ID
        task_output_dir:       
        failed_logs_dir:       
        attempt:     
        output_lines:        （  ）
    """
    try:
        failed_logs_path = Path(failed_logs_dir)
        failed_logs_path.mkdir(parents=True, exist_ok=True)
        
        #         
        task_failed_dir = failed_logs_path / f"{task_id}_attempt_{attempt}"
        if task_failed_dir.exists():
            shutil.rmtree(task_failed_dir)
        
        source_path = Path(task_output_dir)
        if source_path.exists():
            shutil.copytree(source_path, task_failed_dir)
            print(f"  📝         : {task_failed_dir}")
    except Exception as e:
        print(f"  ⚠️           : {e}")


def main():
    """   """
    import argparse
    #         
        #         
    overall_start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description=" CSV      ID         ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    :
  #   4       CSV      
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --workers 4 --config experiments/configs/carla/config_close_gpt-5.yaml
  
  #       （   X11      ）
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --workers 4 --config experiments/configs/carla/config_close_gpt-5.yaml --headless
  
  #       
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --workers 4 --config experiments/configs/carla/config_close_gpt-5.yaml --max-steps 50
  
  #      +       
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --workers 4 --config experiments/configs/carla/config_close_gpt-5.yaml --headless --max-steps 50
  
  #     （     ，  API    ）
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --sequential --config experiments/configs/carla/config_close_gpt-5.yaml --headless
  
  #      +       
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --sequential --config experiments/configs/carla/config_close_gpt-5.yaml --headless --max-steps 50
  
  #         （       benchmark  ）
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --workers 4 --config experiments/configs/carla/config_close_gpt-5.yaml --skip-completed
  
  #      +         
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --sequential --config experiments/configs/carla/config_close_gpt-5.yaml --skip-completed
  
  #        
  python -m scripts.carla.run_benchmark --csv "experiments/csv/carla/Spatial-Annotation-carla.csv" --task carla00200 --config experiments/configs/carla/config_close_gpt-5.yaml
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="    ID CSV    "
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="        (  : 4)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/carla/config_close_gpt-5.yaml",
        help="       (  : experiments/configs/carla/config_close_gpt-5.yaml)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="      "
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="      (  : outputs)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="       (   CloudRendering   ，   X11      )"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="    （         ，     ）    API        "
    )
    
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="        （       benchmark           ）"
    )
    
    parser.add_argument(
        "--only-false",
        action="store_true",
        default=True,
        help="    ：   Completed   null/   （true=   ，false=        ，   ）",
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="   CSV      （   Completed  ），    “   ”    ",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help=(
            "      JSON      API/    null         ；"
            "      (success:false)       2"
        ),
    )

    parser.add_argument(
        "--wait-model-api",
        type=float,
        default=900.0,
        help=(
            "            OpenAI-compatible API      ；"
            "  base_url   localhost/127.0.0.1        900，  0   "
        ),
    )

    parser.add_argument(
        "--wait-model-api-interval",
        type=float,
        default=5.0,
        help="   API         ，   5",
    )
    
    parser.add_argument(
        "--outputs-true-dir",
        type=str,
        default="outputs_true",
        help="          (  : outputs_true)"
    )
    
    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default=None,
        help="   ai2thor_bench：         （    --outputs-true-dir）"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="   carla benchmark：       ID（  carla00200）"
    )
    
    args = parser.parse_args()
    
    #    ai2thor_bench   ：--outputs-completed-dir    --outputs-true-dir    
    if args.outputs_completed_dir:
        args.outputs_true_dir = args.outputs_completed_dir
    
    #   CSV      
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV     : {csv_path}")
        sys.exit(1)
    
    #           
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌        : {config_path}")
        sys.exit(1)
    
    #  CSV    ID（      ）
    if args.task:
        task_ids = [args.task.strip()]
        only_pending = False
        print(f"📋      : {task_ids[0]}")
    else:
        print(f"📋    CSV      ID: {csv_path}")
        # --all-tasks：   Completed；     Completed   null/    （   true   false）
        only_pending = not args.all_tasks
        task_ids = read_task_ids_from_csv(str(csv_path), only_pending=only_pending)

    if not task_ids:
        print("❌ CSV         ID")
        if only_pending:
            print(
                "💡   :       Completed    true/false，     --all-tasks     "
            )
        sys.exit(1)

    print(
        f"✓    {len(task_ids)}    "
        + (" (  Completed   null/ ，  )" if only_pending else "")
    )
    print(f"   5   : {task_ids[:5]}")
    if len(task_ids) > 5:
        print(f"  ... (  {len(task_ids)}    )")
    
    #            
    if args.skip_completed:
        print(f"\n🔍         （    benchmark  ）...")
        completed_tasks = find_completed_tasks(args.output_dir)
        
        if completed_tasks:
            print(f"✓    {len(completed_tasks)}        （ JSON  ）")
            #          
            original_count = len(task_ids)
            task_ids = [tid for tid in task_ids if tid not in completed_tasks]
            skipped_count = original_count - len(task_ids)
            
            if skipped_count > 0:
                print(f"✓     {skipped_count}        ")
                print(f"✓    {len(task_ids)}       ")
                if len(completed_tasks) <= 10:
                    print(f"        : {sorted(completed_tasks)}")
                else:
                    completed_list = sorted(list(completed_tasks))
                    print(f"          （ 10 ）: {completed_list[:10]}...")
                    print(f"  （  {len(completed_tasks)}        ）")
                print(f"\n💡   ：  PNG    JSON           ，     ")
            else:
                print(f"✓         ，    ")
                sys.exit(0)
        else:
            print(f"✓          ，       ")
            print(f"💡   ：  PNG    JSON           ，     ")
    
    if not task_ids:
        print("❌         ")
        sys.exit(0)
    
    #       
    if args.sequential:
        print(f"\n{'=' * 80}")
        print(f"🚀           ")
        print(f"{'=' * 80}")
        print(f"    : {len(task_ids)}")
        print(f"    :   （     ）")
        print(f"    : {config_path}")
        if args.headless:
            print(f"    :     (CloudRendering)")
        print(f"{'=' * 80}\n")
        
        #       
        actual_config_path = config_path
        temp_config_file = None
        
        if args.headless:
            print("🖥️         (CloudRendering)")
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
            print(f"✓          : {actual_config_path}\n")
        
        #         （       ）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_output_dir = os.path.join(args.output_dir, f"benchmark_sequential_{timestamp}")
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        #         
        failed_logs_dir = os.path.join(benchmark_output_dir, "failed_logs")
        os.makedirs(failed_logs_dir, exist_ok=True)
        
        #         （           ）
        task_logs_dir = Path(benchmark_output_dir) / "task_logs"
        task_logs_dir.mkdir(parents=True, exist_ok=True)
        
        #   outputs_true  
        outputs_true_path = Path(args.outputs_true_dir)
        outputs_true_path.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        copied_to_true = 0
        
        #            （        ）
        task_records = []
        
        #      
        if HAS_TQDM:
            task_iterator = tqdm(task_ids, desc="    ", unit="task", ncols=100, 
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            task_iterator = task_ids
        
        for idx, task_id in enumerate(task_iterator, 1):
            if HAS_TQDM:
                #        ，      
                task_iterator.set_description(f"    : {task_id}")
            else:
                print(f"\n{'=' * 80}")
                print(f"📋    {idx}/{len(task_ids)}: {task_id}")
                print(f"{'=' * 80}")
            
            #     ：          JSON（API/   ）   ；    (success:false)   
            task_success = False  #         （CSV true）
            eval_run_done = False  #            JSON
            last_attempt_output_dir = None
            task_start_time = time.time()
            task_log_content = []
            task_status = "failed"
            final_attempt = 0
            
            for attempt in range(1, args.max_retries + 1):
                if attempt > 1:
                    if HAS_TQDM:
                        tqdm.write(f"  🔄    {task_id} (  {attempt}/{args.max_retries}  )")
                    else:
                        print(f"  🔄    {task_id} (  {attempt}/{args.max_retries}  )")
                    time.sleep(2)  #      2 
                
                #         
                task_output_dir = os.path.join(benchmark_output_dir, task_id)
                if attempt > 1:
                    task_output_dir = os.path.join(benchmark_output_dir, f"{task_id}_retry_{attempt}")
                os.makedirs(task_output_dir, exist_ok=True)
                last_attempt_output_dir = task_output_dir
                
                #     
                cmd = [
                    sys.executable,
                    "-m",
                "scripts.carla.work.run_task",
                    "--config", str(actual_config_path),
                    "--tasks", task_id,
                    "--output-dir", task_output_dir
                ]
                
                if args.max_steps:
                    cmd.extend(["--max-steps", str(args.max_steps)])

                try:
                    if not wait_for_local_model_api(
                        actual_config_path,
                        args.wait_model_api,
                        args.wait_model_api_interval,
                    ):
                        raise RuntimeError("Model API tunnel unavailable before task attempt")

                    attempt_start_time = time.time()
                    
                    #     ，          （     + UTF-8，   Win GBK     ）
                    result = _run_capture_utf8(cmd, timeout=None)
                    attempt_duration = time.time() - attempt_start_time
                    
                    #            
                    attempt_log_content = []
                    output_lines = []
                    if result.stdout:
                        output_lines.extend(result.stdout.splitlines())
                        attempt_log_content.append(f"===    {attempt} - STDOUT ===\n")
                        attempt_log_content.append("\n".join(result.stdout.splitlines()))
                        attempt_log_content.append("\n")
                    if result.stderr:
                        output_lines.extend(result.stderr.splitlines())
                        attempt_log_content.append(f"===    {attempt} - STDERR ===\n")
                        attempt_log_content.append("\n".join(result.stderr.splitlines()))
                        attempt_log_content.append("\n")
                    
                    #             
                    attempt_log_content.append(f"===    {attempt} -      ===\n")
                    attempt_log_content.append(f"   : {result.returncode}\n")
                    attempt_log_content.append(f"    : {attempt_duration:.2f}  \n")
                    attempt_log_content.append(f"{'=' * 80}\n\n")
                    
                    #          （    ）
                    task_log_content.extend(attempt_log_content)
                    
                    #            
                    attempt_log = "\n".join(attempt_log_content)
                    
                    #       JSON      （     ）；      JSON   
                    if check_eval_run_finished(task_output_dir):
                        eval_run_done = True
                        result_info = read_result_status_info(find_result_json(task_output_dir))
                        m_ok = model_task_succeeded(result_info)
                        csv_status = decide_csv_status(result_info, task_success=m_ok)
                        task_success = csv_status == "true"
                        if csv_status == "true":
                            task_status = "success"
                        elif csv_status == "false":
                            task_status = "model_fail"
                        else:
                            task_status = "incomplete"
                        final_attempt = attempt

                        log_label = (
                            "success"
                            if csv_status == "true"
                            else ("eval_false" if csv_status == "false" else "incomplete")
                        )
                        save_task_log(task_id, attempt_log, task_logs_dir, attempt, log_label)

                        if csv_status == "true":
                            if HAS_TQDM:
                                tqdm.write(f"  ✅    {task_id}      (   {attempt}  )")
                            else:
                                print(f"  ✅    {task_id}      (   {attempt}  )")
                            if copy_to_outputs_true(task_id, task_output_dir, str(outputs_true_path)):
                                copied_to_true += 1
                                if HAS_TQDM:
                                    tqdm.write(f"  📦      {args.outputs_true_dir}")
                                else:
                                    print(f"  📦      {args.outputs_true_dir}")
                            successful += 1
                        elif csv_status == "false":
                            if HAS_TQDM:
                                tqdm.write(
                                    f"  📋    {task_id}：     ，CSV -> false (   {attempt}  )"
                                )
                            else:
                                print(
                                    f"  📋    {task_id}：     ，CSV -> false (   {attempt}  )"
                                )
                        else:
                            if HAS_TQDM:
                                tqdm.write(
                                    f"  ⚠️    {task_id}：        ，CSV -> null (   {attempt}  )"
                                )
                            else:
                                print(
                                    f"  ⚠️    {task_id}：        ，CSV -> null (   {attempt}  )"
                                )
                            failed += 1

                        task_meta = read_task_metadata(task_id)
                        actual_actions, actual_count = parse_action_sequence(
                            result_info.get("action_sequence")
                        )
                        extra_fields = {
                            "instruction": task_meta.get("instruction"),
                            "golden_action": task_meta.get("golden_action"),
                            "golden_actions_count": task_meta.get("golden_actions_count"),
                            "actual_actions": actual_actions,
                            "actual_actions_count": actual_count,
                            "token_prompt": result_info.get("prompt_tokens"),
                            "token_completion": result_info.get("completion_tokens"),
                            "token_total": result_info.get("total_tokens"),
                            "failure_reason": result_info.get("fail_reason"),
                            "failure_type": result_info.get("failure_type"),
                        }
                        if update_csv_task_record(
                            csv_path, task_id, status=csv_status, extra_fields=extra_fields
                        ):
                            if HAS_TQDM:
                                tqdm.write(f"  ✓ CSV   : {task_id} -> {csv_status} (+   )")
                            else:
                                print(f"  ✓ CSV   : {task_id} -> {csv_status} (+   )")

                        if should_retry_csv_status(csv_status, attempt, args.max_retries):
                            eval_run_done = False
                            if HAS_TQDM:
                                tqdm.write(
                                    f"  🔄    {task_id}   API/    null，     "
                                    f"({attempt + 1}/{args.max_retries})"
                                )
                            else:
                                print(
                                    f"  🔄    {task_id}   API/    null，     "
                                    f"({attempt + 1}/{args.max_retries})"
                                )
                            continue
                        break
                    else:
                        #            
                        save_task_log(task_id, attempt_log, task_logs_dir, attempt, "failed")
                        final_attempt = attempt
                        
                        if HAS_TQDM:
                            tqdm.write(f"  ⚠️     {task_id}         JSON   (   {attempt}/{args.max_retries})")
                        else:
                            print(f"  ⚠️     {task_id}         JSON   (   {attempt}/{args.max_retries})")
                        
                        if attempt == args.max_retries:
                            #         ，      
                            save_failed_log(task_id, task_output_dir, failed_logs_dir, attempt, output_lines)
                except KeyboardInterrupt:
                    if HAS_TQDM:
                        tqdm.write(f"\n⚠️     ，    {idx-1}/{len(task_ids)}    ")
                    else:
                        print(f"\n⚠️     ，    {idx-1}/{len(task_ids)}    ")
                    #          
                    task_log_content.append(f"\n⚠️     \n")
                    save_task_log(task_id, "\n".join(task_log_content), task_logs_dir, final_attempt or 1, "interrupted")
                    raise
                except Exception as e:
                    attempt_duration = time.time() - attempt_start_time
                    error_msg = str(e)
                    if HAS_TQDM:
                        tqdm.write(f"  ❌    {task_id}      (   {attempt}): {e}")
                    else:
                        print(f"  ❌    {task_id}      (   {attempt}): {e}")
                    
                    #         
                    attempt_log_content = []
                    attempt_log_content.append(f"===    {attempt} -      ===\n")
                    attempt_log_content.append(f"  : {error_msg}\n")
                    attempt_log_content.append(f"    : {attempt_duration:.2f}  \n")
                    attempt_log_content.append(f"{'=' * 80}\n\n")
                    
                    #          
                    task_log_content.extend(attempt_log_content)
                    
                    #            
                    attempt_log = "\n".join(attempt_log_content)
                    save_task_log(task_id, attempt_log, task_logs_dir, attempt, "failed")
                    final_attempt = attempt
                    
                    if attempt == args.max_retries:
                        #         ，      
                        save_failed_log(task_id, task_output_dir, failed_logs_dir, attempt, [error_msg])
            
            if not eval_run_done:
                failed += 1
                if HAS_TQDM:
                    tqdm.write(
                        f"  ❌    {task_id}        JSON (    {args.max_retries}  )，CSV   /   null"
                    )
                else:
                    print(
                        f"  ❌    {task_id}        JSON (    {args.max_retries}  )，CSV   /   null"
                    )

                #     JSON：   CSV   null，      
                result_json = find_result_json(last_attempt_output_dir) if last_attempt_output_dir else None
                result_info = read_result_status_info(result_json)
                task_meta = read_task_metadata(task_id)
                actual_actions, actual_count = parse_action_sequence(result_info.get("action_sequence"))
                csv_status = decide_csv_status(result_info, task_success=False)
                fail_reason = result_info.get("fail_reason") or "Task failed after max retries"
                extra_fields = {
                    "instruction": task_meta.get("instruction"),
                    "golden_action": task_meta.get("golden_action"),
                    "golden_actions_count": task_meta.get("golden_actions_count"),
                    "actual_actions": actual_actions,
                    "actual_actions_count": actual_count,
                    "token_prompt": result_info.get("prompt_tokens"),
                    "token_completion": result_info.get("completion_tokens"),
                    "token_total": result_info.get("total_tokens"),
                    "failure_reason": fail_reason,
                    "failure_type": result_info.get("failure_type"),
                }
                update_csv_task_record(csv_path, task_id, status=csv_status, extra_fields=extra_fields)
            
            #     ，     （            ，        ）
            task_duration = time.time() - task_start_time
            
            #             
            task_records.append({
                'task_id': task_id,
                'status': task_status,
                'attempts': final_attempt,
                'duration': task_duration,
                'success': task_success,
                'log_path': None  #             
            })
            
            #          
            if HAS_TQDM:
                task_iterator.set_postfix({
                    '  ': successful,
                    '  ': failed,
                    '   ': copied_to_true
                })
        
        #         
        if temp_config_file and os.path.exists(temp_config_file.name):
            try:
                os.unlink(temp_config_file.name)
            except Exception as e:
                print(f"⚠️           : {e}")
        
        #       
        summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
        total_duration = sum(record['duration'] for record in task_records)
        avg_duration = total_duration / len(task_records) if task_records else 0
        
        with open(summary_log_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"          \n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV  : {csv_path}\n")
            f.write(f"    : {config_path}\n")
            f.write(f"    : {benchmark_output_dir}\n")
            f.write(f"    :   \n")
            if args.headless:
                f.write(f"    :    \n")
            if args.max_steps:
                f.write(f"    : {args.max_steps}\n")
            f.write(f"\n{'=' * 80}\n")
            f.write(f"    \n")
            f.write(f"{'=' * 80}\n")
            f.write(f"    : {len(task_ids)}\n")
            f.write(f"  : {successful} ({successful/len(task_ids)*100:.1f}%)\n")
            f.write(f"  : {failed} ({failed/len(task_ids)*100:.1f}%)\n")
            f.write(f"     {args.outputs_true_dir}: {copied_to_true}\n")
            f.write(f"     : {total_duration:.2f}   ({total_duration/60:.2f}   )\n")
            f.write(f"      : {avg_duration:.2f}  \n")
            f.write(f"\n{'=' * 80}\n")
            f.write(f"    \n")
            f.write(f"{'=' * 80}\n\n")
            
            #      
            successful_tasks = [r for r in task_records if r['success']]
            failed_tasks = [r for r in task_records if not r['success']]
            
            if successful_tasks:
                f.write(f"     ({len(successful_tasks)}  ):\n")
                f.write(f"{'-' * 80}\n")
                for record in successful_tasks:
                    f.write(f"  ✅ {record['task_id']}: {record['duration']:.2f} , {record['attempts']}   \n")
                f.write(f"\n")
            
            if failed_tasks:
                f.write(f"     ({len(failed_tasks)}  ):\n")
                f.write(f"{'-' * 80}\n")
                for record in failed_tasks:
                    f.write(f"  ❌ {record['task_id']}: {record['duration']:.2f} , {record['attempts']}   \n")
                f.write(f"\n")
            
            f.write(f"{'=' * 80}\n")
            f.write(f"      \n")
            f.write(f"{'=' * 80}\n")
            for i, record in enumerate(task_records, 1):
                status_icon = "✅" if record['success'] else "❌"
                f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                       f"  : {record['status']:10s} | "
                       f"  : {record['attempts']:2d} | "
                       f"  : {record['duration']:8.2f} ")
                if record['log_path']:
                    f.write(f" |   : {record['log_path']}")
                f.write(f"\n")
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"      \n")
            f.write(f"{'=' * 80}\n")
            f.write(f"      : {task_logs_dir}\n")
            f.write(f"    : {summary_log_path}\n")
            if failed > 0:
                f.write(f"      : {failed_logs_dir}\n")
        
        #     
        print(f"\n{'=' * 80}")
        print(f"🎉       ")
        #        
        overall_duration = time.time() - overall_start_time
        error_rate = (failed / len(task_ids) * 100) if len(task_ids) > 0 else 0
        print(f"{'=' * 80}")
        print(f"    : {len(task_ids)}")
        print(f"  : {successful}")
        print(f"  : {failed}")
        print(f"     {args.outputs_true_dir}: {copied_to_true}")
        print(f"    : {benchmark_output_dir}")
        print(f"      : {task_logs_dir}")
        print(f"    : {summary_log_path}")
        if failed > 0:
            print(f"      : {failed_logs_dir}")
        print(f"{'=' * 80}\n")
        
        sys.exit(0 if failed == 0 else 1)
    
    #       （    ）
    #       ：    ，        
    actual_config_path = config_path
    temp_config_file = None
    
    if args.headless:
        print("🖥️         (CloudRendering)")
        #         
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        #    env     
        if "env" not in config_data:
            config_data["env"] = {}
        
        #    platform   CloudRendering
        config_data["env"]["platform"] = "CloudRendering"
        
        #         
        temp_config_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            delete=False,
            encoding='utf-8'
        )
        yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
        temp_config_file.close()
        
        actual_config_path = Path(temp_config_file.name)
        print(f"✓          : {actual_config_path}")
    
    #       
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    #         
    task_logs_dir = Path(benchmark_output_dir) / "task_logs"
    task_logs_dir.mkdir(parents=True, exist_ok=True)
    
    #   outputs_true  
    outputs_true_path = Path(args.outputs_true_dir)
    outputs_true_path.mkdir(parents=True, exist_ok=True)
    
    # CSV   （      ）
    csv_lock = Lock()
    
    print(f"\n{'=' * 80}")
    print(f"🚀           ")
    print(f"{'=' * 80}")
    print(f"    : {len(task_ids)}")
    print(f"     : {args.workers}")
    print(f"    : {actual_config_path}")
    print(f"    : {benchmark_output_dir}")
    if args.headless:
        print(f"    :     (CloudRendering)")
    print(f"{'=' * 80}\n")
    
    #            （   ）
    def execute_task(task_id: str) -> dict:
        """           ，            CSV"""
        task_start_time = time.time()
        task_log_content = []
        task_status = "failed"
        final_attempt = 0
        task_success = False
        eval_run_done = False
        task_output_dir = os.path.join(benchmark_output_dir, task_id)
        
        for attempt in range(1, args.max_retries + 1):
            if attempt > 1:
                task_output_dir = os.path.join(benchmark_output_dir, f"{task_id}_retry_{attempt}")
            os.makedirs(task_output_dir, exist_ok=True)
            
            #     
            cmd = [
                sys.executable,
                "-m",
                "scripts.carla.work.run_task",
                "--config", str(actual_config_path),
                "--tasks", task_id,
                "--output-dir", task_output_dir
            ]
            
            if args.max_steps:
                cmd.extend(["--max-steps", str(args.max_steps)])
            
            try:
                if not wait_for_local_model_api(
                    actual_config_path,
                    args.wait_model_api,
                    args.wait_model_api_interval,
                ):
                    raise RuntimeError("Model API tunnel unavailable before task attempt")

                attempt_start_time = time.time()
                result = _run_capture_utf8(cmd, timeout=None)
                attempt_duration = time.time() - attempt_start_time
                
                #            
                attempt_log_content = []
                if result.stdout:
                    attempt_log_content.append(f"===    {attempt} - STDOUT ===\n")
                    attempt_log_content.append(result.stdout)
                    attempt_log_content.append("\n")
                if result.stderr:
                    attempt_log_content.append(f"===    {attempt} - STDERR ===\n")
                    attempt_log_content.append(result.stderr)
                    attempt_log_content.append("\n")
                
                attempt_log_content.append(f"===    {attempt} -      ===\n")
                attempt_log_content.append(f"   : {result.returncode}\n")
                attempt_log_content.append(f"    : {attempt_duration:.2f}  \n")
                attempt_log_content.append(f"{'=' * 80}\n\n")
                
                #          （        ）
                task_log_content.extend(attempt_log_content)
                
                #            
                attempt_log = "".join(attempt_log_content)
                
                if check_eval_run_finished(task_output_dir):
                    eval_run_done = True
                    result_info = read_result_status_info(find_result_json(task_output_dir))
                    m_ok = model_task_succeeded(result_info)
                    csv_status = decide_csv_status(result_info, task_success=m_ok)
                    task_success = csv_status == "true"
                    if csv_status == "true":
                        task_status = "success"
                    elif csv_status == "false":
                        task_status = "model_fail"
                    else:
                        task_status = "incomplete"
                    final_attempt = attempt
                    log_label = (
                        "success"
                        if csv_status == "true"
                        else ("eval_false" if csv_status == "false" else "incomplete")
                    )
                    save_task_log(task_id, attempt_log, task_logs_dir, attempt, log_label)

                    if csv_status == "true":
                        if copy_to_outputs_true(task_id, task_output_dir, str(outputs_true_path)):
                            print(
                                f"  ✅ {task_id}      (   {attempt}  ) ->      {args.outputs_true_dir}"
                            )
                    elif csv_status == "false":
                        print(f"  📋 {task_id}：      -> false (   {attempt}  )")
                    else:
                        print(f"  ⚠️ {task_id}：         -> null (   {attempt}  )")

                    task_meta = read_task_metadata(task_id)
                    actual_actions, actual_count = parse_action_sequence(
                        result_info.get("action_sequence")
                    )
                    extra_fields = {
                        "instruction": task_meta.get("instruction"),
                        "golden_action": task_meta.get("golden_action"),
                        "golden_actions_count": task_meta.get("golden_actions_count"),
                        "actual_actions": actual_actions,
                        "actual_actions_count": actual_count,
                        "token_prompt": result_info.get("prompt_tokens"),
                        "token_completion": result_info.get("completion_tokens"),
                        "token_total": result_info.get("total_tokens"),
                        "failure_reason": result_info.get("fail_reason"),
                        "failure_type": result_info.get("failure_type"),
                    }
                    update_csv_task_record(
                        csv_path, task_id, status=csv_status, extra_fields=extra_fields, lock=csv_lock
                    )

                    if should_retry_csv_status(csv_status, attempt, args.max_retries):
                        eval_run_done = False
                        print(
                            f"  🔄 {task_id}：API/    null，     "
                            f"({attempt + 1}/{args.max_retries})"
                        )
                        continue
                    break
                else:
                    attempt_status = "failed"
                    final_attempt = attempt
                    save_task_log(task_id, attempt_log, task_logs_dir, attempt, attempt_status)
            except Exception as e:
                attempt_duration = time.time() - attempt_start_time
                attempt_log_content = []
                attempt_log_content.append(f"===    {attempt} -      ===\n")
                attempt_log_content.append(f"  : {str(e)}\n")
                attempt_log_content.append(f"    : {attempt_duration:.2f}  \n")
                attempt_log_content.append(f"{'=' * 80}\n\n")
                task_log_content.extend(attempt_log_content)
                
                attempt_status = "failed"
                final_attempt = attempt
                #            
                attempt_log = "".join(attempt_log_content)
                save_task_log(task_id, attempt_log, task_logs_dir, attempt, attempt_status)

        #       JSON：   CSV（   null）
        if not eval_run_done:
            result_info = read_result_status_info(find_result_json(task_output_dir))
            task_meta = read_task_metadata(task_id)
            actual_actions, actual_count = parse_action_sequence(result_info.get("action_sequence"))
            csv_status = decide_csv_status(result_info, task_success=False)
            fail_reason = result_info.get("fail_reason") or "Task failed after max retries"
            extra_fields = {
                "instruction": task_meta.get("instruction"),
                "golden_action": task_meta.get("golden_action"),
                "golden_actions_count": task_meta.get("golden_actions_count"),
                "actual_actions": actual_actions,
                "actual_actions_count": actual_count,
                "token_prompt": result_info.get("prompt_tokens"),
                "token_completion": result_info.get("completion_tokens"),
                "token_total": result_info.get("total_tokens"),
                "failure_reason": fail_reason,
                "failure_type": result_info.get("failure_type"),
            }
            update_csv_task_record(csv_path, task_id, status=csv_status, extra_fields=extra_fields, lock=csv_lock)

        #     ，     
        task_duration = time.time() - task_start_time
        
        #     （        ，               ）
        infra_fail = (not eval_run_done) or (task_status == "incomplete")
        return {
            'task_id': task_id,
            'status': task_status,
            'attempts': final_attempt,
            'duration': task_duration,
            'success': task_success,
            'eval_run_done': eval_run_done,
            'infra_fail': infra_fail,
            'log_path': None  #             
        }
    
    #            
    task_records = []
    successful = 0
    failed = 0
    copied_to_true = 0
    exit_code = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            #       
            future_to_task = {
                executor.submit(execute_task, task_id): task_id
                for task_id in task_ids
            }
            
            #          
            if HAS_TQDM:
                task_iterator = tqdm(as_completed(future_to_task), total=len(task_ids), 
                                    desc="    ", unit="task", ncols=100,
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                task_iterator = as_completed(future_to_task)
            
            #        
            for future in task_iterator:
                try:
                    result = future.result()
                    task_records.append(result)
                    
                    if result['success']:
                        successful += 1
                        copied_to_true += 1
                    elif result.get('infra_fail'):
                        failed += 1
                    
                    if HAS_TQDM:
                        task_iterator.set_postfix({
                            '  ': successful,
                            '  ': failed,
                            '   ': copied_to_true
                        })
                except Exception as e:
                    print(f"\n❌       : {e}")
                    failed += 1
    
    except KeyboardInterrupt:
        print("\n⚠️     ")
        exit_code = 1
    else:
        exit_code = 0 if failed == 0 else 1
    
    #       
    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    total_duration = sum(record['duration'] for record in task_records)
    avg_duration = total_duration / len(task_records) if task_records else 0
    
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"          \n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV  : {csv_path}\n")
        f.write(f"    : {config_path}\n")
        f.write(f"    : {benchmark_output_dir}\n")
        f.write(f"    :    (workers: {args.workers})\n")
        if args.headless:
            f.write(f"    :    \n")
        if args.max_steps:
            f.write(f"    : {args.max_steps}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"    \n")
        f.write(f"{'=' * 80}\n")
        f.write(f"    : {len(task_ids)}\n")
        f.write(f"  : {successful} ({successful/len(task_ids)*100:.1f}%)\n")
        f.write(f"  : {failed} ({failed/len(task_ids)*100:.1f}%)\n")
        f.write(f"     {args.outputs_true_dir}: {copied_to_true}\n")
        f.write(f"     : {total_duration:.2f}   ({total_duration/60:.2f}   )\n")
        f.write(f"      : {avg_duration:.2f}  \n")
        f.write(f"      : {args.max_retries}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"    \n")
        f.write(f"{'=' * 80}\n\n")
        
        #      
        successful_tasks = [r for r in task_records if r['success']]
        failed_tasks = [r for r in task_records if not r['success']]
        
        if successful_tasks:
            f.write(f"     ({len(successful_tasks)}  ):\n")
            f.write(f"{'-' * 80}\n")
            for record in successful_tasks:
                f.write(f"  ✅ {record['task_id']}: {record['duration']:.2f} , {record['attempts']}   \n")
            f.write(f"\n")
        
        if failed_tasks:
            f.write(f"     ({len(failed_tasks)}  ):\n")
            f.write(f"{'-' * 80}\n")
            for record in failed_tasks:
                f.write(f"  ❌ {record['task_id']}: {record['duration']:.2f} , {record['attempts']}   \n")
            f.write(f"\n")
        
        f.write(f"{'=' * 80}\n")
        f.write(f"      \n")
        f.write(f"{'=' * 80}\n")
        for i, record in enumerate(task_records, 1):
            status_icon = "✅" if record['success'] else "❌"
            f.write(f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                   f"  : {record['status']:10s} | "
                   f"  : {record['attempts']:2d} | "
                   f"  : {record['duration']:8.2f} ")
            if record['log_path']:
                f.write(f" |   : {record['log_path']}")
            f.write(f"\n")
        
        f.write(f"\n{'=' * 80}\n")
        f.write(f"      \n")
        f.write(f"{'=' * 80}\n")
        f.write(f"      : {task_logs_dir}\n")
        f.write(f"    : {summary_log_path}\n")
    
    #     
    overall_duration = time.time() - overall_start_time
    error_rate = (failed / len(task_ids) * 100) if len(task_ids) > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"🎉       ")
    print(f"{'=' * 80}")
    print(f"    : {len(task_ids)}")
    print(f"  : {successful}")
    print(f"   : {error_rate:.2f}%")

    print(f"  : {failed}")
    print(f"     {args.outputs_true_dir}: {copied_to_true}")
    print(f"     : {overall_duration:.2f}   ({overall_duration/60:.2f}   )")

    print(f"    : {benchmark_output_dir}")
    print(f"      : {task_logs_dir}")
    print(f"    : {summary_log_path}")
    print(f"{'=' * 80}\n")
    
    #         
    if temp_config_file and os.path.exists(temp_config_file.name):
        try:
            os.unlink(temp_config_file.name)
            print(f"✓          ")
        except Exception as e:
            print(f"⚠️           : {e}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
