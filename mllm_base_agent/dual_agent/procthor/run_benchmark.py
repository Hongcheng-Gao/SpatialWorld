#!/usr/bin/env python3
"""ProcTHOR dual-agent benchmark runner.

     AI2-THOR   `spatial-planning/dual_agent/run_benchmark.py`   CSV      ：
-    CSV，   Completed   （null）  
-         Completed: true / false / null（  ）
-      ：golden_action / instruction / token_total / failure_reason /
  actual_actions / golden_actions_count / actual_actions_count
-    --save-name / --headless / --workers / --sequential / --skip-completed
-      `python -m mllm_base_agent.dual_agent.procthor.main`，   ProcTHOR     （ai2thor    ）

  ：
    python -m mllm_base_agent.dual_agent.procthor.run_benchmark \\
        --csv "experiments/csv/procthor/Spatial-Annotation-procthor-Gemini-2.5-pro.csv" \\
        --config experiments/configs/procthor/dual/config_close_gpt-5.yaml \\
        --save-name Gemini-2.5-Pro --headless --workers 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from dotenv import load_dotenv


_THIS_FILE = Path(__file__).resolve()
DUAL_AGENT_DIR = _THIS_FILE.parent
PROJECT_ROOT = _THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_FOLDER_ROOT = DUAL_AGENT_DIR / "task_mutil_procthor"
DEFAULT_BENCHMARK_OUTPUT_DIR = DUAL_AGENT_DIR / "benchmark_outputs"
DEFAULT_OUTPUTS_COMPLETED_DIR = DUAL_AGENT_DIR / "outputs_completed"

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed, fall back to simple progress. (pip install tqdm)")


INFLIGHT_TASKS_LOCK = Lock()
INFLIGHT_TASKS: Set[str] = set()

CSV_EXTRA_FIELDS = [
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


# ---------------------------------------------------------------------------
# CSV helpers (aligned with spatial-planning/run_csv_benchmark.py)
# ---------------------------------------------------------------------------


def normalize_task_id(task_id: str) -> str:
    """Normalize task id, e.g. procthor_201 -> procthor201."""
    if "_" in task_id and task_id.startswith("procthor_"):
        return task_id.replace("procthor_", "procthor", 1)
    if "_" in task_id and task_id.startswith("ai2thor_"):
        return task_id.replace("ai2thor_", "ai2thor", 1)
    return task_id


def read_task_ids_from_csv(csv_path: Path, only_null: bool = True) -> List[str]:
    """Read task IDs from CSV; when only_null, skip rows whose Completed is true/false."""
    task_ids: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        completed_idx = None
        if header:
            for idx, col in enumerate(header):
                if col.strip().lower() in ("completed", "status"):
                    completed_idx = idx
                    break
        for row in reader:
            if not row or not row[0].strip():
                continue
            task_id = row[0].strip()
            if task_id == "Task ID":
                continue
            if only_null and completed_idx is not None and len(row) > completed_idx:
                status = row[completed_idx].strip().lower()
                if status in ("true", "false"):
                    continue
            task_ids.append(task_id)
    return task_ids


def count_csv_status(csv_path: Path) -> Dict[str, int]:
    counts = {"total": 0, "true": 0, "false": 0, "null": 0}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return counts
        completed_idx = None
        for idx, col in enumerate(header):
            if col.strip().lower() in ("completed", "status"):
                completed_idx = idx
                break
        if completed_idx is None:
            return counts
        for row in reader:
            if not row or not row[0].strip():
                continue
            counts["total"] += 1
            status = row[completed_idx].strip().lower() if len(row) > completed_idx else ""
            if status == "true":
                counts["true"] += 1
            elif status == "false":
                counts["false"] += 1
            else:
                counts["null"] += 1
    return counts


def deduplicate_task_ids(task_ids: List[str]) -> Tuple[List[str], List[str]]:
    seen: Set[str] = set()
    unique: List[str] = []
    duplicates: List[str] = []
    for tid in task_ids:
        norm = normalize_task_id(tid)
        if norm in seen:
            duplicates.append(tid)
            continue
        seen.add(norm)
        unique.append(tid)
    return unique, duplicates


def update_csv_task_record(
    csv_path: Path,
    task_id: str,
    status: Optional[str],
    extra_fields: Optional[Dict[str, Any]] = None,
    lock: Optional[Lock] = None,
) -> bool:
    """Update the row matching task_id. status=None means keep empty (null)."""
    if lock:
        lock.acquire()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return False
        header = rows[0]

        def _col_idx(name: str) -> int:
            for idx, col in enumerate(header):
                if col.strip().lower() == name.lower():
                    return idx
            header.append(name)
            for i in range(1, len(rows)):
                while len(rows[i]) < len(header):
                    rows[i].append("")
            return len(header) - 1

        completed_idx = _col_idx("Completed")
        extra_idx_map: Dict[str, int] = {}
        if extra_fields:
            for key in extra_fields:
                extra_idx_map[key] = _col_idx(key)

        normalized_target = normalize_task_id(task_id)
        updated = False
        for i in range(1, len(rows)):
            if not rows[i] or not rows[i][0].strip():
                continue
            row_tid = rows[i][0].strip()
            if normalize_task_id(row_tid) != normalized_target and row_tid != task_id:
                continue
            while len(rows[i]) < len(header):
                rows[i].append("")

            if status is None:
                new_status = ""
            else:
                new_status = str(status)
            if rows[i][completed_idx] != new_status:
                rows[i][completed_idx] = new_status
                updated = True

            if extra_fields:
                for key, idx in extra_idx_map.items():
                    value = extra_fields.get(key)
                    if value is None:
                        continue
                    text = str(value)
                    if rows[i][idx] != text:
                        rows[i][idx] = text
                        updated = True
            break

        if updated:
            backup_path = csv_path.with_suffix(csv_path.suffix + ".backup")
            if not backup_path.exists():
                shutil.copy2(csv_path, backup_path)
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        return updated
    except Exception as e:
        print(f"  ⚠️  Update CSV failed ({task_id}): {e}")
        return False
    finally:
        if lock:
            lock.release()


# ---------------------------------------------------------------------------
# Task metadata & logs
# ---------------------------------------------------------------------------


def load_task_metadata(task_id: str) -> Dict[str, Any]:
    """Load metadata from mllm_base_agent/dual_agent/procthor/task_mutil_procthor/<task_id>/task.json."""
    info: Dict[str, Any] = {
        "instruction": None,
        "golden_action_count": None,
        "golden_action_text": None,
    }
    candidates = [
        TASK_FOLDER_ROOT / normalize_task_id(task_id) / "task.json",
        TASK_FOLDER_ROOT / task_id / "task.json",
        PROJECT_ROOT / "tasks" / normalize_task_id(task_id) / "task.json",
    ]
    task_json: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if not task_json:
        return info

    try:
        with open(task_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        info["instruction"] = data.get("instruction") or data.get("target_description") or data.get("task_name")
        golden = data.get("golden_actions")
        actions: List[str] = []
        steps_int: Optional[int] = None
        if isinstance(golden, dict):
            s = golden.get("steps")
            if isinstance(s, int):
                steps_int = int(s)
            raw = golden.get("actions") or []
            if isinstance(raw, list):
                actions = [str(a).strip() for a in raw if str(a).strip()]
        elif isinstance(golden, list):
            actions = [str(a).strip() for a in golden if str(a).strip()]
        elif isinstance(golden, str):
            actions = [a.strip() for a in golden.split(",") if a.strip()]

        if actions:
            info["golden_action_text"] = " | ".join(actions)
        # CSV golden_actions_count：  task.json   golden_actions.steps   （     steps，          ）
        if steps_int is not None:
            info["golden_action_count"] = steps_int
        elif actions:
            non_done = [a for a in actions if a.upper() != "DONE"]
            info["golden_action_count"] = len(non_done) if non_done else len(actions)
    except Exception as e:
        print(f"  ⚠️  Failed to load task metadata for {task_id}: {e}")
    return info


def extract_token_stats_from_text(text: str) -> Dict[str, Optional[int]]:
    """Sum prompt/completion/total tokens printed to stdout/stderr."""
    if not text:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    pattern = r"(?:['\"])?{key}(?:['\"])?\s*[:=]\s*(\d+)"
    prompt = [int(x) for x in re.findall(pattern.format(key="prompt_tokens"), text, flags=re.IGNORECASE)]
    completion = [int(x) for x in re.findall(pattern.format(key="completion_tokens"), text, flags=re.IGNORECASE)]
    total = [int(x) for x in re.findall(pattern.format(key="total_tokens"), text, flags=re.IGNORECASE)]
    prompt_sum = sum(prompt) if prompt else None
    completion_sum = sum(completion) if completion else None
    if total:
        total_sum = sum(total)
    elif prompt_sum is not None and completion_sum is not None:
        total_sum = prompt_sum + completion_sum
    else:
        total_sum = None
    return {
        "prompt_tokens": prompt_sum,
        "completion_tokens": completion_sum,
        "total_tokens": total_sum,
    }


def save_task_log(task_id: str, log_content: str, logs_dir: Path, attempt: int, status: str) -> None:
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{task_id}_attempt_{attempt}_{status}.log"
        path = logs_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(log_content)
    except Exception as e:
        print(f"  ⚠️  Save task log failed ({task_id}): {e}")


def build_csv_extra_fields(
    task_metadata: Dict[str, Any],
    actual_actions_info: Dict[str, Any],
    token_stats: Dict[str, Optional[int]],
    failure_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "golden_action": task_metadata.get("golden_action_text"),
        "instruction": task_metadata.get("instruction"),
        "token_total": token_stats.get("total_tokens"),
        "failure_reason": failure_reason,
        "actual_actions": actual_actions_info.get("actual_action_text"),
        "golden_actions_count": task_metadata.get("golden_action_count"),
        "actual_actions_count": actual_actions_info.get("actual_action_count"),
    }


def write_missing_result_diagnostic(
    task_output_dir: str,
    task_id: str,
    return_code: Optional[int],
    stdout_text: str,
    stderr_text: str,
    error_text: Optional[str] = None,
) -> None:
    try:
        path = Path(task_output_dir) / "run_error.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if return_code is not None:
                f.write(f"Return code: {return_code}\n")
            if error_text:
                f.write(f"Error: {error_text}\n")
            f.write("=" * 80 + "\n")
            f.write("No result JSON found in task output directory.\n\n")
            if stdout_text:
                f.write("=== STDOUT (tail) ===\n")
                f.write(stdout_text[-10000:])
                f.write("\n\n")
            if stderr_text:
                f.write("=== STDERR (tail) ===\n")
                f.write(stderr_text[-10000:])
    except Exception as e:
        print(f"  ⚠️  Failed to write diagnostic: {e}")


# ---------------------------------------------------------------------------
# Result JSON parsing
# ---------------------------------------------------------------------------


def find_result_json(task_output_dir: str) -> Optional[Path]:
    task_path = Path(task_output_dir)
    if not task_path.exists():
        return None
    # prefer dual_episode_*.json (rich); fall back to log.json
    dual_candidates = sorted(task_path.glob("dual_episode_*.json"))
    if dual_candidates:
        return dual_candidates[-1]
    log_file = task_path / "log.json"
    if log_file.exists():
        return log_file
    nested_dual = sorted(task_path.rglob("dual_episode_*.json"))
    if nested_dual:
        return nested_dual[-1]
    nested_log = sorted(task_path.rglob("log.json"))
    if nested_log:
        return nested_log[-1]
    return None


def read_result_status_info(result_json: Optional[Path]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "has_result_json": bool(result_json and result_json.exists()),
        "task_result": None,
        "failure_type": None,
        "fail_reason": None,
        "agent_1_steps": 0,
        "agent_2_steps": 0,
        "communication_events": 0,
        "turn_count": 0,
    }
    if not result_json or not result_json.exists():
        return info

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if result_json.name == "log.json":
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            tr = metadata.get("task_result")
            if tr in ("success", "failure"):
                info["task_result"] = tr
            info["failure_type"] = metadata.get("failure_type")
            info["fail_reason"] = metadata.get("fail_reason")
            info["agent_1_steps"] = metadata.get("agent_1_steps", 0) or 0
            info["agent_2_steps"] = metadata.get("agent_2_steps", 0) or 0
            info["communication_events"] = metadata.get("communication_events", 0) or 0
        else:
            success = data.get("success")
            if success is True:
                info["task_result"] = "success"
            elif success is False:
                info["task_result"] = "failure"
            info["failure_type"] = data.get("failure_type")
            info["fail_reason"] = data.get("fail_reason")
            info["agent_1_steps"] = data.get("agent_1_steps", 0) or 0
            info["agent_2_steps"] = data.get("agent_2_steps", 0) or 0
            info["turn_count"] = data.get("turn_count", 0) or 0
            info["communication_events"] = len(data.get("communication_history", []) or [])
    except Exception as e:
        info["fail_reason"] = f"Failed to parse result JSON: {e}"
    return info


def extract_actual_actions(result_json: Optional[Path]) -> Dict[str, Any]:
    empty = {"actual_action_count": None, "actual_action_text": None}
    if not result_json or not result_json.exists():
        return empty
    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        trajectory = data.get("trajectory")
        if not isinstance(trajectory, list):
            # log.json schema: messages assistant -> action_executed
            trajectory = []
            for msg in data.get("messages", []) if isinstance(data, dict) else []:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    ae = msg.get("action_executed")
                    if ae:
                        trajectory.append({"action_string": ae})

        actions: List[str] = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            a = (entry.get("action_string") or "").strip()
            if a:
                actions.append(a)
        return {
            "actual_action_count": len(actions),
            "actual_action_text": " | ".join(actions) if actions else None,
        }
    except Exception as e:
        print(f"  ⚠️  Failed to extract actual actions: {e}")
        return empty


# ---------------------------------------------------------------------------
# Failure classification (ported from ai2thor run_benchmark.py)
# ---------------------------------------------------------------------------


def _sanitized_log_for_classification(task_log_content: str) -> str:
    if not task_log_content:
        return ""
    kept: List[str] = []
    for line in task_log_content.splitlines():
        lo = line.lower()
        if "parsed action" in lo and ("✓" in line or "success" in lo):
            continue
        if "api request attempt" in lo:
            continue
        if "using custom platform parameter" in lo and "cloudrendering" in lo:
            continue
        kept.append(line)
    return "\n".join(kept)


def _fail_reason_indicates_model_failure(fr: str) -> bool:
    if not fr or "failed to parse result json" in fr:
        return False
    if "consecutive" in fr and "action failure" in fr:
        return True
    patterns = (
        "reached maximum", "maximum global step", "max_steps", "max steps", "max step",
        "step limit", "reached max steps", "agent_1 reached max", "agent_2 reached max",
        "agent_1    ", "agent_2    ", "      ", "    ", "    ",
        "task incomplete", "blocked or exhausted", "both agents exhausted", "cannot continue",
        "premature done", "done but", "model claimed done", "model output invalid action",
        "no action available", "final evaluation on terminal state failed", "exceeded maximum",
        "maximum step", "refused to continue", "model determined",
    )
    if any(p in fr for p in patterns):
        return True
    if "final evaluation" in fr and "fail" in fr:
        return True
    return False


def _fail_reason_indicates_external_api(fr: str) -> bool:
    if not fr:
        return False
    patterns = (
        "rate limit", "too many requests", "resource exhausted", "resourceexhausted",
        "quota exceeded", "invalid api key", "api key invalid", "authentication failed",
        "unauthorized", "forbidden", "connection refused", "connection reset", "econnrefused",
        "network is unreachable", "name or service not known", "read timed out",
        "request timed out", "handshake failure", "ssl error", "certificate",
        "google.api_core.exceptions", "internal server error", "httpstatuserror",
        "bad gateway", "service unavailable", "not available in your region",
    )
    if any(p in fr for p in patterns):
        return True
    for token in (" 429", " 502", " 503", " 401", " 403", ": 429", ": 502", ": 503", ": 401", ": 403"):
        if token in fr:
            return True
    if fr.strip() in ("429", "502", "503", "401", "403"):
        return True
    if "timeout" in fr and any(x in fr for x in ("request", "read", "connect", "api", "http", "socket")):
        return True
    return False


def _fail_reason_indicates_external_env(fr: str) -> bool:
    if not fr:
        return False
    patterns = (
        "env_crash", "environment exception", "unity crashed", "unity crash", "gpu process",
        "segmentation fault", "cuda error", "vulkan error", "could not create",
        "could not connect to display",
    )
    if any(p in fr for p in patterns):
        return True
    if "cloudrendering" in fr and any(x in fr for x in ("error", "fail", "exception", "unable", "crash")):
        return True
    return False


def _log_indicates_external_api(log_low: str) -> bool:
    if not log_low:
        return False
    patterns = (
        "rate limit", "too many requests", "resourceexhausted", "quota exceeded",
        "429 too many", "502 bad gateway", "503 service unavailable", "401 unauthorized",
        "403 forbidden", "connection refused", "connection reset", "econnrefused",
        "read timed out", "request timed out", "api error", "invalid api key",
        "authentication failed", "google.api_core.exceptions", "internal server error",
        "httpstatuserror", "not available in your region",
    )
    if any(p in log_low for p in patterns):
        return True
    if "timeout" in log_low and any(x in log_low for x in ("request", "read", "connect", "api", "http", "socket", "urllib", "httpx")):
        return True
    return False


def _log_indicates_external_env(log_low: str) -> bool:
    if not log_low:
        return False
    patterns = (
        "env_crash", "environment exception", "unity crashed", "segmentation fault",
        "gpu process crashed", "could not connect to display",
    )
    if any(p in log_low for p in patterns):
        return True
    if "cloudrendering" in log_low and any(x in log_low for x in ("error", "fail", "exception", "unable", "crash")):
        return True
    return False


def _log_indicates_parse_or_action_error(log_low: str) -> Optional[str]:
    if not log_low:
        return None
    parse_markers = (
        "failed to parse", "parse error", "json decode error", "jsondecodeerror",
        "invalid json", "could not parse model output", "malformed json",
    )
    if any(p in log_low for p in parse_markers):
        return "parse_error"
    action_markers = ("invalid action", "no action available", "action error")
    if any(p in log_low for p in action_markers):
        return "action_error"
    return None


def determine_failure_reason(task_log_content: str = "", result_json: Optional[Path] = None) -> str:
    result_info = read_result_status_info(result_json)
    failure_type = result_info.get("failure_type")
    if failure_type:
        return failure_type
    fail_reason = (result_info.get("fail_reason") or "").strip()
    fr_low = fail_reason.lower()
    task_result = result_info.get("task_result")

    if fr_low:
        if "failed to parse result json" in fr_low:
            return "external_error"
        if _fail_reason_indicates_external_api(fr_low):
            return "api_error"
        if _fail_reason_indicates_external_env(fr_low):
            return "env_error"
        if _fail_reason_indicates_model_failure(fr_low):
            return "model_error"

    san = _sanitized_log_for_classification(task_log_content)
    log_low = san.lower()

    if log_low:
        if _log_indicates_external_api(log_low):
            return "api_error"
        if _log_indicates_external_env(log_low):
            return "env_error"
        pa = _log_indicates_parse_or_action_error(log_low)
        if pa:
            return pa

    if task_result == "failure":
        return "model_error"
    return "external_error"


def decide_csv_status_from_result(result_info: Dict[str, Any], fallback_failure_type: str) -> Optional[str]:
    task_result = result_info.get("task_result")
    fail_reason_raw = result_info.get("fail_reason") or ""
    fail_reason = fail_reason_raw.lower()
    json_ft = result_info.get("failure_type")

    if task_result == "success":
        return "true"

    if json_ft in ("api_error", "env_error", "external_error"):
        return None

    if fail_reason:
        if "failed to parse result json" in fail_reason:
            return None
        if _fail_reason_indicates_model_failure(fail_reason):
            return "false"
        if _fail_reason_indicates_external_api(fail_reason) or _fail_reason_indicates_external_env(fail_reason):
            return None

    failure_type = json_ft or fallback_failure_type
    if failure_type in ("api_error", "env_error", "external_error"):
        return None
    if failure_type in ("parse_error", "action_error", "model_error"):
        return "false"

    if task_result == "failure":
        return "false"
    return None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def copytree_no_delete(source: Path, dest: Path) -> Path:
    final = dest
    if final.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final = dest.parent / f"{dest.name}_{suffix}"
    shutil.copytree(source, final)
    return final


def copy_to_outputs_completed(task_id: str, task_output_dir: str, outputs_completed_dir: str) -> bool:
    try:
        src = Path(task_output_dir)
        if not src.exists():
            return False
        dest = Path(outputs_completed_dir) / task_id
        copytree_no_delete(src, dest)
        return True
    except Exception as e:
        print(f"  ❌ Copy to outputs_completed failed ({task_id}): {e}")
        return False


def save_failed_snapshot(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int = 1) -> None:
    try:
        src = Path(task_output_dir)
        if not src.exists():
            return
        dst_root = Path(failed_logs_dir)
        dst_root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        copytree_no_delete(src, dst_root / f"{task_id}_attempt_{attempt}_{ts}")
    except Exception as e:
        print(f"  ⚠️  Error saving failed snapshot: {e}")


def find_completed_tasks(output_dir: str, save_name: Optional[str]) -> Set[str]:
    completed: Set[str] = set()
    root = Path(output_dir)
    if not root.exists():
        return completed
    prefixes = ["dual_benchmark_", "dual_benchmark_sequential_"]
    if save_name:
        prefixes.append(f"{save_name}_")
    dirs = [d for d in root.iterdir() if d.is_dir() and any(d.name.startswith(p) for p in prefixes)]
    dirs.sort(key=lambda x: x.name, reverse=True)
    for bdir in dirs:
        for task_dir in bdir.iterdir():
            if not task_dir.is_dir():
                continue
            if task_dir.name in {"task_logs", "failed_logs"}:
                continue
            if find_result_json(str(task_dir)):
                completed.add(normalize_task_id(task_dir.name))
    return completed


def prepare_benchmark_config(config_path: Path, headless: bool) -> Tuple[Path, Optional[str]]:
    """Possibly generate a temp config with headless / platform overrides."""
    if not headless:
        return config_path, None
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("env", {})
    display = os.environ.get("DISPLAY", "").strip()
    x_display_cfg = str(data["env"].get("x_display", "")).strip()
    runtime_x = x_display_cfg or display
    if runtime_x:
        data["env"]["x_display"] = runtime_x
        platform = data["env"].get("platform")
        if isinstance(platform, str) and platform.lower() == "cloudrendering":
            data["env"]["platform"] = None
    else:
        data["env"]["platform"] = "CloudRendering"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(data, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name), tmp.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="ProcTHOR dual-agent benchmark runner (CSV, null/true/false)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m mllm_base_agent.dual_agent.procthor.run_benchmark --csv "experiments/csv/procthor/Spatial-Annotation-procthor-Gemini-2.5-pro.csv" \\
      --config experiments/configs/procthor/dual/config_close_gpt-5.yaml --save-name Gemini-2.5-Pro --headless --workers 5
""",
    )
    parser.add_argument("--csv", type=str, default=None, help="   CSV   ")
    parser.add_argument("--workers", type=int, default=4, help="   worker  ")
    parser.add_argument("--config", type=str, default="experiments/configs/procthor/dual/config_close_gpt-5.yaml", help="        ")
    parser.add_argument("--task", type=str, default=None, help="      task_id")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="            ")
    parser.add_argument("--max-steps", type=int, default=None, help="         ")
    parser.add_argument("--switch-interval", type=int, default=None, help="          ")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_BENCHMARK_OUTPUT_DIR),
        help="benchmark      （   dual_agent/benchmark_outputs）",
    )
    parser.add_argument(
        "--collaboration-mode",
        type=str,
        default="alternating",
        choices=["alternating", "sequential"],
        help="        ",
    )
    parser.add_argument("--headless", action="store_true", help="  headless     （CloudRendering   xvfb DISPLAY）")
    parser.add_argument("--sequential", action="store_true", help="      ")
    parser.add_argument("--skip-completed", action="store_true", help="       ")
    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_COMPLETED_DIR),
        help="         （   dual_agent/outputs_completed）",
    )
    parser.add_argument("--save-name", type=str, default=None, help="benchmark       ")
    parser.add_argument("--agent1", type=str, default=None, help="Agent 1       ")
    parser.add_argument("--agent2", type=str, default=None, help="Agent 2       ")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    csv_path: Optional[Path] = None
    if args.csv:
        p = Path(args.csv)
        if not p.is_absolute():
            # prefer dual_agent-local, then project root
            for c in [DUAL_AGENT_DIR / args.csv, PROJECT_ROOT / args.csv, p]:
                if c.exists():
                    p = c
                    break
        if not p.exists():
            print(f"❌ CSV file not found: {p}")
            return 1
        csv_path = p

    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    elif args.tasks:
        task_ids = [tid.strip() for tid in args.tasks if tid.strip()]
        print(f"📋 Explicit task list: {task_ids[:5]}")
    elif csv_path:
        print(f"📋 Reading task IDs from CSV: {csv_path}")
        task_ids = read_task_ids_from_csv(csv_path, only_null=True)
        if not task_ids:
            print("❌ No task IDs with Completed=null in CSV")
            return 1
        print(f"✓ Found {len(task_ids)} tasks with Completed=null")
    else:
        print("❌ Provide one of --csv / --task / --tasks")
        return 1

    print(f"  First 5: {task_ids[:5]}")
    if len(task_ids) > 5:
        print(f"  ... (total {len(task_ids)} tasks)")

    if args.skip_completed:
        print("\n🔍 Checking completed tasks from previous dual benchmark dirs...")
        completed = find_completed_tasks(args.output_dir, args.save_name)
        if completed:
            original = len(task_ids)
            task_ids = [tid for tid in task_ids if normalize_task_id(tid) not in completed]
            print(f"✓ Found {len(completed)} completed; skipping {original - len(task_ids)}")
            if not task_ids:
                print("✓ All tasks completed")
                return 0
        else:
            print("✓ No completed tasks found")

    task_ids, duplicates = deduplicate_task_ids(task_ids)
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicated task IDs; duplicates skipped")
    if not task_ids:
        print("❌ No tasks to run")
        return 0

    actual_config_path, temp_config = prepare_benchmark_config(config_path, args.headless)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_name:
        prefix = args.save_name
    else:
        prefix = "dual_benchmark_sequential" if args.sequential else "dual_benchmark"
    benchmark_output_dir = Path(args.output_dir) / f"{prefix}_{timestamp}"
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
    print(f"Collaboration mode: {args.collaboration_mode}")
    if args.max_steps:
        print(f"Max steps: {args.max_steps}")
    if args.switch_interval:
        print(f"Switch interval: {args.switch_interval}")
    if args.headless:
        print("Headless: enabled")
    print(f"{'=' * 80}\n")

    def execute_task(task_id: str) -> Dict[str, Any]:
        task_start = time.time()
        task_log_parts: List[str] = []
        task_status = "failed_external"
        task_success = False
        copied_to_completed = False
        task_metadata = load_task_metadata(task_id)
        token_stats = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        actual_actions_info = {"actual_action_count": None, "actual_action_text": None}
        failure_reason_detail: Optional[str] = None
        result_info = {
            "has_result_json": False,
            "task_result": None,
            "failure_type": None,
            "fail_reason": None,
            "agent_1_steps": 0,
            "agent_2_steps": 0,
            "communication_events": 0,
            "turn_count": 0,
        }

        normalized = normalize_task_id(task_id)
        with INFLIGHT_TASKS_LOCK:
            if normalized in INFLIGHT_TASKS:
                return {
                    "task_id": task_id,
                    "status": "failed_external",
                    "duration": time.time() - task_start,
                    "success": False,
                    "failure_reason": "Task already running (dedupe protection)",
                    "copied_to_completed": False,
                    "task_result": None,
                    "agent_1_steps": 0,
                    "agent_2_steps": 0,
                    "communication_events": 0,
                    "turn_count": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "golden_actions_count": task_metadata.get("golden_action_count"),
                    "actual_actions_count": None,
                }
            INFLIGHT_TASKS.add(normalized)

        benchmark_task_output_dir = benchmark_output_dir / task_id
        benchmark_task_output_dir.mkdir(parents=True, exist_ok=True)
        stdout_text = ""
        stderr_text = ""
        result_json: Optional[Path] = None

        try:
            cmd = [
                sys.executable,
                "-m",
                "mllm_base_agent.dual_agent.procthor.main",
                "--config",
                str(actual_config_path),
                "--tasks",
                normalized,
                "--output-dir",
                str(benchmark_task_output_dir),
                "--collaboration-mode",
                args.collaboration_mode,
            ]
            if args.agent1:
                cmd.extend(["--agent1", args.agent1])
            if args.agent2:
                cmd.extend(["--agent2", args.agent2])
            if args.max_steps:
                cmd.extend(["--max-steps", str(args.max_steps)])
            if args.switch_interval:
                cmd.extend(["--switch-interval", str(args.switch_interval)])
            if args.headless:
                cmd.append("--headless")

            exec_start = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=None,
            )
            exec_dur = time.time() - exec_start
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
            task_log_parts.append(f"Command: {' '.join(cmd)}\n")
            task_log_parts.append(f"Exit code: {result.returncode}\n")
            task_log_parts.append(f"Duration: {exec_dur:.2f}s\n")
            task_log_parts.append("=" * 80 + "\n\n")

            task_log = "".join(task_log_parts)
            token_stats = extract_token_stats_from_text(task_log)
            result_json = find_result_json(str(benchmark_task_output_dir))
            result_info = read_result_status_info(result_json)
            actual_actions_info = extract_actual_actions(result_json)
            failure_reason_detail = result_info.get("fail_reason")

            if result_info.get("task_result") == "success":
                task_success = True
                task_status = "success"
                save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                copied_to_completed = copy_to_outputs_completed(
                    task_id, str(benchmark_task_output_dir), str(outputs_completed_path)
                )
                if csv_path:
                    extra = build_csv_extra_fields(
                        task_metadata, actual_actions_info, token_stats, failure_reason_detail
                    )
                    update_csv_task_record(csv_path, task_id, status="true", extra_fields=extra, lock=csv_lock)
                print(f"  ✅ {task_id} success")
            else:
                failure_reason = determine_failure_reason(task_log, result_json)
                csv_status = decide_csv_status_from_result(result_info, failure_reason)
                failure_reason_detail = result_info.get("fail_reason") or failure_reason
                extra = build_csv_extra_fields(
                    task_metadata, actual_actions_info, token_stats, failure_reason_detail
                )
                save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                if benchmark_task_output_dir.exists():
                    save_failed_snapshot(task_id, str(benchmark_task_output_dir), str(failed_logs_dir), 1)

                if result_json:
                    if failure_reason in ("api_error", "env_error", "external_error") and csv_status != "false":
                        task_status = "failed_external"
                        print(f"  ⚠️  {task_id} external failure -> null")
                    else:
                        task_status = "failed_model"
                        print(f"  ❌ {task_id} model failure -> {csv_status}")

                    if csv_status == "false":
                        copied_to_completed = copy_to_outputs_completed(
                            task_id, str(benchmark_task_output_dir), str(outputs_completed_path)
                        )
                    if csv_path:
                        update_csv_task_record(
                            csv_path, task_id, status=csv_status, extra_fields=extra, lock=csv_lock
                        )
                else:
                    task_status = "failed_external"
                    failure_reason_detail = "No result JSON produced"
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir),
                        task_id,
                        result.returncode,
                        stdout_text,
                        stderr_text,
                        error_text=failure_reason_detail,
                    )
                    if csv_path:
                        update_csv_task_record(csv_path, task_id, status=None, extra_fields=extra, lock=csv_lock)
                    print(f"  ⚠️  {task_id} no result JSON -> null")

        except KeyboardInterrupt:
            task_status = "interrupted"
            save_task_log(task_id, "".join(task_log_parts), task_logs_dir, 1, "interrupted")
            raise
        except Exception as e:
            error_msg = str(e)
            task_log_parts.append("=== Exception ===\n")
            task_log_parts.append(f"Error: {error_msg}\n")
            task_log_parts.append("=" * 80 + "\n\n")
            task_log = "".join(task_log_parts)
            save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
            if benchmark_task_output_dir.exists():
                save_failed_snapshot(task_id, str(benchmark_task_output_dir), str(failed_logs_dir), 1)

            result_json = find_result_json(str(benchmark_task_output_dir))
            result_info = read_result_status_info(result_json)
            token_stats = extract_token_stats_from_text(task_log)
            actual_actions_info = extract_actual_actions(result_json)
            failure_reason = determine_failure_reason(task_log, result_json)
            failure_reason_detail = result_info.get("fail_reason") or error_msg
            csv_status = decide_csv_status_from_result(result_info, failure_reason)
            extra = build_csv_extra_fields(task_metadata, actual_actions_info, token_stats, failure_reason_detail)

            if failure_reason in ("parse_error", "action_error", "model_error"):
                task_status = "failed_model"
                if csv_status == "false":
                    copied_to_completed = copy_to_outputs_completed(
                        task_id, str(benchmark_task_output_dir), str(outputs_completed_path)
                    )
            else:
                task_status = "failed_external"
                if not result_json:
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir), task_id, None, "", "", error_text=error_msg
                    )
            if csv_path:
                update_csv_task_record(csv_path, task_id, status=csv_status, extra_fields=extra, lock=csv_lock)
            print(f"  ❌ {task_id} exception: {error_msg}")
        finally:
            with INFLIGHT_TASKS_LOCK:
                INFLIGHT_TASKS.discard(normalized)

        duration = time.time() - task_start
        return {
            "task_id": task_id,
            "status": task_status,
            "duration": duration,
            "success": task_success,
            "failure_reason": failure_reason_detail,
            "copied_to_completed": copied_to_completed,
            "task_result": result_info.get("task_result"),
            "agent_1_steps": result_info.get("agent_1_steps", 0),
            "agent_2_steps": result_info.get("agent_2_steps", 0),
            "communication_events": result_info.get("communication_events", 0),
            "turn_count": result_info.get("turn_count", 0),
            "prompt_tokens": token_stats.get("prompt_tokens"),
            "completion_tokens": token_stats.get("completion_tokens"),
            "total_tokens": token_stats.get("total_tokens"),
            "golden_actions_count": task_metadata.get("golden_action_count"),
            "actual_actions_count": actual_actions_info.get("actual_action_count"),
        }

    records: List[Dict[str, Any]] = []
    successful = 0
    failed_model = 0
    failed_external = 0
    copied_count = 0
    exit_code = 0

    try:
        if args.sequential:
            iterator = tqdm(task_ids, desc="Tasks", unit="task", ncols=100) if HAS_TQDM else task_ids
            for idx, tid in enumerate(iterator, 1):
                if not HAS_TQDM:
                    print(f"\n{'=' * 80}\n📋 Task {idx}/{len(task_ids)}: {tid}\n{'=' * 80}")
                res = execute_task(tid)
                records.append(res)
                if res["success"]:
                    successful += 1
                elif res["status"] == "failed_model":
                    failed_model += 1
                else:
                    failed_external += 1
                if res.get("copied_to_completed"):
                    copied_count += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_map = {pool.submit(execute_task, tid): tid for tid in task_ids}
                iterator = (
                    tqdm(as_completed(future_map), total=len(task_ids), desc="Tasks", unit="task", ncols=100)
                    if HAS_TQDM else as_completed(future_map)
                )
                for fut in iterator:
                    res = fut.result()
                    records.append(res)
                    if res["success"]:
                        successful += 1
                    elif res["status"] == "failed_model":
                        failed_model += 1
                    else:
                        failed_external += 1
                    if res.get("copied_to_completed"):
                        copied_count += 1
                    if HAS_TQDM:
                        iterator.set_postfix(
                            {"ok": successful, "model": failed_model, "external": failed_external, "copied": copied_count}
                        )
    except KeyboardInterrupt:
        print("\n⚠️ User interrupt")
        exit_code = 1
    else:
        exit_code = 0 if (failed_model + failed_external) == 0 else 1
    finally:
        if temp_config and os.path.exists(temp_config):
            try:
                os.unlink(temp_config)
            except Exception:
                pass

    total_dur = sum(r["duration"] for r in records)
    avg_dur = total_dur / len(records) if records else 0.0
    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ProcTHOR dual-agent benchmark summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if csv_path:
            f.write(f"CSV: {csv_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Actual config: {actual_config_path}\n")
        f.write(f"Output dir: {benchmark_output_dir}\n")
        f.write(f"Mode: {'sequential' if args.sequential else f'parallel (workers: {args.workers})'}\n")
        f.write(f"Collaboration mode: {args.collaboration_mode}\n")
        if args.max_steps:
            f.write(f"Max steps: {args.max_steps}\n")
        if args.switch_interval:
            f.write(f"Switch interval: {args.switch_interval}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total tasks: {len(task_ids)}\n")
        if task_ids:
            f.write(f"Success: {successful} ({successful / len(task_ids) * 100:.1f}%)\n")
            f.write(f"Model failure: {failed_model} ({failed_model / len(task_ids) * 100:.1f}%)\n")
            f.write(f"External failure: {failed_external} ({failed_external / len(task_ids) * 100:.1f}%)\n")
        f.write(f"Copied to outputs_completed: {copied_count}\n")
        f.write(f"Total time: {total_dur:.2f}s ({total_dur / 60:.2f} min)\n")
        f.write(f"Avg time: {avg_dur:.2f}s\n")

    summary_json = {
        "timestamp": timestamp,
        "csv": str(csv_path) if csv_path else None,
        "config": str(config_path),
        "actual_config": str(actual_config_path),
        "output_dir": str(benchmark_output_dir),
        "mode": "sequential" if args.sequential else "parallel",
        "workers": 1 if args.sequential else args.workers,
        "collaboration_mode": args.collaboration_mode,
        "max_steps": args.max_steps,
        "switch_interval": args.switch_interval,
        "headless": args.headless,
        "total_tasks": len(task_ids),
        "successful": successful,
        "failed_model": failed_model,
        "failed_external": failed_external,
        "copied_to_completed": copied_count,
        "duration_seconds": total_dur,
        "avg_duration_seconds": avg_dur,
        "task_records": records,
    }
    with open(benchmark_output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print("🎉 ProcTHOR dual-agent benchmark complete")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Success: {successful}")
    print(f"Model failure: {failed_model}")
    print(f"External failure: {failed_external}")
    print(f"Copied to outputs_completed: {copied_count}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Summary log: {summary_log_path}")

    if csv_path:
        stats = count_csv_status(csv_path)
        total = stats["total"]
        if total:
            print(f"\nCSV status (total {total}):")
            print(f"  true: {stats['true']} ({stats['true'] / total * 100:.1f}%)")
            print(f"  false: {stats['false']} ({stats['false'] / total * 100:.1f}%)")
            print(f"  null: {stats['null']} ({stats['null'] / total * 100:.1f}%)")

    print(f"{'=' * 80}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
