#!/usr/bin/env python3
"""
     benchmark     

       `run_csv_benchmark.py`：
-    CSV     
-     /    
-    skip-completed save-name outputs-completed   
-         CSV

        ：
-         `python -m mllm_base_agent.dual_agent.ai2thor.main`
-    JSON    `mllm_base_agent/dual_agent/ai2thor/main.py`     `dual_episode_*.json`
- benchmark           `mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/.../<task_id>/`（   --output-dir   ）
-             （`env.agent_count: 2`）；          setdefault，         YAML   `env.agent_count: 1`
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import yaml
from dotenv import load_dotenv

#           ，            ，
#                      `config`  ，
#        `dual_agent/config` 
project_root = str(Path(__file__).resolve().parents[3])
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)
dual_agent_dir = Path(__file__).resolve().parent
#        dual_agent    （    ，        cwd）
_DEFAULT_BENCHMARK_OUTPUT_DIR = str(dual_agent_dir / "benchmark_outputs")
_DEFAULT_OUTPUTS_COMPLETED_DIR = str(dual_agent_dir / "outputs_completed")
_DEFAULT_DUAL_AGENT_CONFIG = str((Path(__file__).resolve().parents[3] / "configs" / "ai2thor" / "dual" / "config.yaml").resolve())
_DEFAULT_DUAL_CSV_DIR = Path(project_root) / "experiments" / "csv" / "ai2thor" / "dual"
os.environ.setdefault(
    "AI2THOR_TASKS_ROOT",
    str(Path(project_root) / "data" / "ai2thor" / "dual" / "tasks"),
)

from .config import load_config
from scripts.ai2thor.run_benchmark import (  #        benchmark
    build_csv_extra_fields,
    count_csv_status,
    deduplicate_task_ids,
    extract_token_stats_from_text,
    load_task_metadata,
    read_task_ids_from_csv,
    save_task_log,
    update_csv_task_record,
    write_missing_result_diagnostic,
)

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed, using simple progress. Install: pip install tqdm")


INFLIGHT_TASKS_LOCK = Lock()
INFLIGHT_TASKS: Set[str] = set()


def normalize_task_id(task_id: str) -> str:
    """    task id，   ai2thor_04000 / ai2thor04000      """
    if "_" in task_id and task_id.startswith("ai2thor_"):
        return task_id.replace("ai2thor_", "ai2thor", 1)
    return task_id


def find_result_json(task_output_dir: str) -> Optional[Path]:
    """               JSON """
    task_path = Path(task_output_dir)
    if not task_path.exists():
        return None

    direct_candidates = sorted(task_path.glob("dual_episode_*.json"))
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(task_path.rglob("dual_episode_*.json"))
    if nested_candidates:
        return nested_candidates[-1]

    return None


def resolve_dual_agent_csv_path(csv_arg: Optional[str]) -> Optional[Path]:
    """Resolve dual AI2-THOR CSVs, preferring experiments/csv/ai2thor/dual."""
    if not csv_arg:
        return None

    raw = Path(csv_arg)
    basename = raw.name
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.extend([raw, _DEFAULT_DUAL_CSV_DIR / basename])
    else:
        candidates.extend(
            [
                _DEFAULT_DUAL_CSV_DIR / basename,
                Path(project_root) / raw,
                dual_agent_dir / raw,
                raw,
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def find_completed_tasks(output_dir: str, save_name: Optional[str] = None) -> set:
    """    dual benchmark            """
    completed = set()
    output_path = Path(output_dir)

    if not output_path.exists():
        return completed

    prefixes = ["dual_benchmark_", "dual_benchmark_sequential_"]
    if save_name:
        prefixes.append(f"{save_name}_")

    benchmark_dirs = sorted(
        [
            d
            for d in output_path.iterdir()
            if d.is_dir() and any(d.name.startswith(prefix) for prefix in prefixes)
        ],
        key=lambda x: x.name,
        reverse=True,
    )

    for benchmark_dir in benchmark_dirs:
        for task_dir in benchmark_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if task_dir.name in {"task_logs", "failed_logs"}:
                continue
            if find_result_json(str(task_dir)):
                completed.add(normalize_task_id(task_dir.name))

    return completed


def read_result_status_info(result_json: Optional[Path]) -> Dict[str, Any]:
    """         JSON        """
    info = {
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


def _sanitized_log_for_classification(task_log_content: str) -> str:
    """                   API/         """
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


def _fail_reason_indicates_model_failure(fr_lower: str) -> bool:
    """fail_reason                      →   /    """
    if not fr_lower or "failed to parse result json" in fr_lower:
        return False
    if "consecutive" in fr_lower and "action failure" in fr_lower:
        return True
    patterns = (
        "reached maximum",
        "maximum global step",
        "max_steps",
        "max steps",
        "max step",
        "step limit",
        "reached max steps",
        "agent_1 reached max",
        "agent_2 reached max",
        "agent_1    ",
        "agent_2    ",
        "      ",
        "    ",
        "    ",
        "task incomplete",
        "blocked or exhausted",
        "both agents exhausted",
        "cannot continue",
        "premature done",
        "done but",
        "model claimed done",
        "model output invalid action",
        "no action available",
        "final evaluation on terminal state failed",
        "exceeded maximum",
        "maximum step",
        "refused to continue",
        "model determined",
    )
    if any(p in fr_lower for p in patterns):
        return True
    if "final evaluation" in fr_lower and "fail" in fr_lower:
        return True
    return False


def _fail_reason_indicates_external_api(fr_lower: str) -> bool:
    """fail_reason          API /    /    /    """
    if not fr_lower:
        return False
    patterns = (
        "rate limit",
        "too many requests",
        "resource exhausted",
        "resourceexhausted",
        "quota exceeded",
        "invalid api key",
        "api key invalid",
        "authentication failed",
        "unauthorized",
        "forbidden",
        "connection refused",
        "connection reset",
        "econnrefused",
        "network is unreachable",
        "name or service not known",
        "read timed out",
        "request timed out",
        "handshake failure",
        "ssl error",
        "certificate",
        "google.api_core.exceptions",
        "internal server error",
        "httpstatuserror",
        "bad gateway",
        "service unavailable",
        "not available in your region",
    )
    if any(p in fr_lower for p in patterns):
        return True
    for token in (" 429", " 502", " 503", " 401", " 403", ": 429", ": 502", ": 503", ": 401", ": 403"):
        if token in fr_lower:
            return True
    if fr_lower.strip() in ("429", "502", "503", "401", "403"):
        return True
    if "timeout" in fr_lower and any(
        x in fr_lower for x in ("request", "read", "connect", "api", "http", "socket")
    ):
        return True
    return False


def _fail_reason_indicates_external_env(fr_lower: str) -> bool:
    """fail_reason        /    /      """
    if not fr_lower:
        return False
    patterns = (
        "env_crash",
        "environment exception",
        "unity crashed",
        "unity crash",
        "gpu process",
        "segmentation fault",
        "cuda error",
        "vulkan error",
        "could not create",
        "could not connect to display",
    )
    if any(p in fr_lower for p in patterns):
        return True
    if "cloudrendering" in fr_lower and any(
        x in fr_lower for x in ("error", "fail", "exception", "unable", "crash")
    ):
        return True
    return False


def _log_indicates_external_api(log_lower: str) -> bool:
    if not log_lower:
        return False
    patterns = (
        "rate limit",
        "too many requests",
        "resourceexhausted",
        "quota exceeded",
        "429 too many",
        "502 bad gateway",
        "503 service unavailable",
        "401 unauthorized",
        "403 forbidden",
        "connection refused",
        "connection reset",
        "econnrefused",
        "read timed out",
        "request timed out",
        "api error",
        "invalid api key",
        "authentication failed",
        "google.api_core.exceptions",
        "internal server error",
        "httpstatuserror",
        "not available in your region",
    )
    if any(p in log_lower for p in patterns):
        return True
    if "timeout" in log_lower and any(
        x in log_lower for x in ("request", "read", "connect", "api", "http", "socket", "urllib", "httpx")
    ):
        return True
    return False


def _log_indicates_external_env(log_lower: str) -> bool:
    if not log_lower:
        return False
    patterns = (
        "env_crash",
        "environment exception",
        "unity crashed",
        "segmentation fault",
        "gpu process crashed",
        "could not connect to display",
    )
    if any(p in log_lower for p in patterns):
        return True
    if "cloudrendering" in log_lower and any(
        x in log_lower for x in ("error", "fail", "exception", "unable", "crash")
    ):
        return True
    return False


def _log_indicates_parse_or_action_error(log_lower: str) -> Optional[str]:
    if not log_lower:
        return None
    parse_markers = (
        "failed to parse",
        "parse error",
        "json decode error",
        "jsondecodeerror",
        "invalid json",
        "could not parse model output",
        "malformed json",
    )
    if any(p in log_lower for p in parse_markers):
        return "parse_error"
    action_markers = (
        "invalid action",
        "no action available",
        "action error",
    )
    if any(p in log_lower for p in action_markers):
        return "action_error"
    return None


def determine_failure_reason(task_log_content: str = "", result_json_path: Optional[Path] = None) -> str:
    """      ：JSON.failure_type > fail_reason    >          >      """
    result_info = read_result_status_info(result_json_path)
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
        if "graphrecursionerror" in log_low or "recursion limit" in log_low:
            return "recursion_limit"
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
    """             CSV：true / false / None( ，   ) 

         fail_reason    ，                    failure_type        null 
    """
    task_result = result_info.get("task_result")
    fail_reason_raw = result_info.get("fail_reason") or ""
    fail_reason = fail_reason_raw.lower()
    json_ft = result_info.get("failure_type")

    if task_result == "success":
        return "true"

    #  ：         （  run     env_error）→    /    false；    env_error→false
    if fail_reason and (
        (
            "called with invalid argument" in fail_reason
            and "expected arguments" in fail_reason
        )
        or "invalid action parameters:" in fail_reason
    ):
        return "false"

    #        JSON           ，              
    if json_ft in ("api_error", "env_error", "external_error"):
        return None

    if fail_reason:
        if "failed to parse result json" in fail_reason:
            return None
        if _fail_reason_indicates_model_failure(fail_reason):
            return "false"
        if _fail_reason_indicates_external_api(fail_reason) or _fail_reason_indicates_external_env(
            fail_reason
        ):
            return None

    failure_type = json_ft or fallback_failure_type

    if failure_type in ("api_error", "env_error", "external_error", "recursion_limit"):
        return None
    if failure_type in ("parse_error", "action_error", "model_error"):
        return "false"

    if task_result == "failure":
        return "false"

    return None


def extract_actual_actions(result_json: Optional[Path]) -> Dict[str, Any]:
    """        JSON           """
    empty = {"actual_action_count": None, "actual_action_text": None}
    if not result_json or not result_json.exists():
        return empty

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectory = data.get("trajectory", [])
        if not isinstance(trajectory, list):
            return empty

        actions = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            action_string = (entry.get("action_string") or "").strip()
            if not action_string:
                continue
            actions.append(action_string)

        return {
            "actual_action_count": len(actions) if actions else None,
            "actual_action_text": " | ".join(actions) if actions else None,
        }
    except Exception as e:
        print(f"  ⚠️  Failed to extract actual actions: {e}")
        return empty


def snapshot_task_run_dirs(task_id: str) -> Set[str]:
    """     task        dual_agent      """
    outputs_root = Path(project_root) / "dual_agent" / "outputs"
    normalized = normalize_task_id(task_id)
    if not outputs_root.exists():
        return set()
    return {str(path.resolve()) for path in outputs_root.glob(f"task_{normalized}_*") if path.is_dir()}


def find_new_task_run_dir(task_id: str, before_dirs: Set[str], started_at: float) -> Optional[Path]:
    """         task          """
    outputs_root = Path(project_root) / "dual_agent" / "outputs"
    normalized = normalize_task_id(task_id)
    if not outputs_root.exists():
        return None

    candidates = [path for path in outputs_root.glob(f"task_{normalized}_*") if path.is_dir()]
    if not candidates:
        return None

    new_dirs = [path for path in candidates if str(path.resolve()) not in before_dirs]
    if new_dirs:
        return max(new_dirs, key=lambda x: x.stat().st_mtime)

    recent_dirs = [path for path in candidates if path.stat().st_mtime >= started_at - 1.0]
    if recent_dirs:
        return max(recent_dirs, key=lambda x: x.stat().st_mtime)

    return max(candidates, key=lambda x: x.stat().st_mtime)


def copytree_no_delete(source: Path, dest: Path) -> Path:
    """    ；              ，     """
    final_dest = dest
    if final_dest.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dest = dest.parent / f"{dest.name}_{suffix}"
    shutil.copytree(source, final_dest)
    return final_dest


def copy_to_outputs_completed(task_id: str, task_output_dir: str, outputs_completed_dir: str) -> bool:
    """       ，        """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            print(f"  ⚠️  Source dir not found: {task_output_dir}")
            return False

        dest_path = Path(outputs_completed_dir) / task_id
        copytree_no_delete(source_path, dest_path)
        return True
    except Exception as e:
        print(f"  ❌ Copy to outputs_completed failed ({task_id}): {e}")
        return False


def save_failed_snapshot(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int = 1):
    """          ，        """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            return
        failed_logs_path = Path(failed_logs_dir)
        failed_logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = failed_logs_path / f"{task_id}_attempt_{attempt}_{timestamp}"
        copytree_no_delete(source_path, dest_path)
        print(f"  📝 Failed snapshot saved to: {dest_path}")
    except Exception as e:
        print(f"  ⚠️  Error saving failed snapshot: {e}")


def prepare_benchmark_config(
    config_path: Path,
    headless: bool,
    collaboration_mode: str,
) -> tuple[Path, Optional[str]]:
    """             ，   benchmark      """
    needs_temp = headless or collaboration_mode != "alternating"
    if not needs_temp:
        return config_path, None

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    config_data.setdefault("env", {})
    # benchmark         （  mllm_base_agent.dual_agent.ai2thor.main   ）；YAML    agent_count      
    config_data["env"].setdefault("agent_count", 2)

    if headless:
        config_data["env"]["platform"] = "CloudRendering"

    if "dual_agent" not in config_data:
        config_data["dual_agent"] = {}
    config_data["dual_agent"]["collaboration_mode"] = collaboration_mode

    temp_config_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
    temp_config_file.close()
    return Path(temp_config_file.name), temp_config_file.name


def main():
    """    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dual-agent CSV benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --workers 4 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --sequential --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --tasks ai2thor05001 ai2thor05002 --workers 2 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --collaboration-mode sequential --switch-interval 5
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml --agent1 experiments/configs/ai2thor/dual/config_close_Gemini-3.1-Pro-Preview.yaml --agent2 experiments/configs/ai2thor/dual/config_kimi-a3b.yaml
""",
    )

    parser.add_argument("--csv", type=str, default=None, help="   CSV   ")
    parser.add_argument("--workers", type=int, default=4, help="   worker  ")
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_DUAL_AGENT_CONFIG,
        help="        （   experiments/configs/ai2thor/dual/config_close_gpt-5.yaml，    agent_count=2）",
    )
    parser.add_argument("--task", type=str, default=None, help="      task_id")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="            ",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="              ")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Override per-agent max steps (default: 10 + golden_actions.steps)",
    )
    parser.add_argument(
        "--switch-interval",
        type=int,
        default=None,
        help="          ",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_BENCHMARK_OUTPUT_DIR,
        help="benchmark      （   dual_agent/benchmark_outputs）",
    )
    parser.add_argument(
        "--collaboration-mode",
        type=str,
        default="alternating",
        choices=["alternating", "parallel", "sequential"],
        help="        ",
    )
    parser.add_argument("--headless", action="store_true", help="  CloudRendering     ")
    parser.add_argument("--sequential", action="store_true", help="      ")
    parser.add_argument("--skip-completed", action="store_true", help="       ")
    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default=_DEFAULT_OUTPUTS_COMPLETED_DIR,
        help="         （   dual_agent/outputs_completed）",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="benchmark       ",
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default=None,
        help="Agent 1            ；      agent config，       ",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default=None,
        help="Agent 2            ；   agent1   ，       ",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    csv_path = resolve_dual_agent_csv_path(args.csv)
    if csv_path and not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    elif args.tasks:
        task_ids = [task_id.strip() for task_id in args.tasks if task_id.strip()]
        print(f"📋 Explicit task list: {task_ids[:5]}")
    elif csv_path:
        print(f"📋 Reading task IDs from CSV: {csv_path}")
        task_ids = read_task_ids_from_csv(str(csv_path), only_null=True)
        if not task_ids:
            print("❌ No task IDs with Completed=null in CSV")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks with Completed=null")
    else:
        print(f"📋 Loading tasks from config: {config_path}")
        config = load_config(str(config_path))
        task_ids = config.get_all_task_names()
        if not task_ids:
            print("❌ No tasks found in config")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks from config")

    print(f"  First 5: {task_ids[:5]}")
    if len(task_ids) > 5:
        print(f"  ... (total {len(task_ids)} tasks)")

    if args.skip_completed:
        print("\n🔍 Checking completed tasks from previous dual benchmark dirs...")
        completed_tasks = find_completed_tasks(args.output_dir, args.save_name)
        if completed_tasks:
            original_count = len(task_ids)
            task_ids = [
                task_id
                for task_id in task_ids
                if normalize_task_id(task_id) not in completed_tasks
            ]
            skipped_count = original_count - len(task_ids)
            print(f"✓ Found {len(completed_tasks)} completed tasks")
            print(f"✓ Skipping {skipped_count} tasks")
            if not task_ids:
                print("✓ All tasks completed, nothing to run")
                sys.exit(0)
        else:
            print("✓ No completed tasks found, will run all")

    unique_task_ids, duplicate_task_ids = deduplicate_task_ids(task_ids)
    if duplicate_task_ids:
        print(f"⚠️  Found {len(duplicate_task_ids)} duplicated task IDs; duplicates will be skipped")
    task_ids = unique_task_ids

    if not task_ids:
        print("❌ No tasks to run")
        sys.exit(0)

    actual_config_path, temp_config_path = prepare_benchmark_config(
        config_path=config_path,
        headless=args.headless,
        collaboration_mode=args.collaboration_mode,
    )

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
    print(f"🚀 Starting {'sequential' if args.sequential else 'parallel'} dual-agent benchmark")
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
        print("Headless: enabled (CloudRendering)")
    print(f"{'=' * 80}\n")

    def execute_task(task_id: str) -> Dict[str, Any]:
        task_start_time = time.time()
        task_log_parts: List[str] = []
        task_status = "failed_external"
        task_success = False
        copied_to_completed = False
        task_metadata = load_task_metadata(task_id)
        token_stats = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        actual_actions_info = {"actual_action_count": None, "actual_action_text": None}
        failure_reason_detail = None
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

        normalized_task_id = normalize_task_id(task_id)
        with INFLIGHT_TASKS_LOCK:
            if normalized_task_id in INFLIGHT_TASKS:
                return {
                    "task_id": task_id,
                    "status": "failed_external",
                    "attempts": 1,
                    "duration": time.time() - task_start_time,
                    "success": False,
                    "golden_actions_count": task_metadata.get("golden_action_count"),
                    "actual_actions_count": actual_actions_info.get("actual_action_count"),
                    "golden_action": task_metadata.get("golden_action_text"),
                    "actual_actions": actual_actions_info.get("actual_action_text"),
                    "instruction": task_metadata.get("instruction"),
                    "prompt_tokens": token_stats["prompt_tokens"],
                    "completion_tokens": token_stats["completion_tokens"],
                    "total_tokens": token_stats["total_tokens"],
                    "failure_reason": "Task already running (dedupe protection)",
                    "task_result": None,
                    "agent_1_steps": 0,
                    "agent_2_steps": 0,
                    "communication_events": 0,
                    "turn_count": 0,
                    "copied_to_completed": False,
                }
            INFLIGHT_TASKS.add(normalized_task_id)

        benchmark_task_output_dir = benchmark_output_dir / task_id
        benchmark_task_output_dir.mkdir(parents=True, exist_ok=True)
        stdout_text = ""
        stderr_text = ""
        result_json = None

        try:
            before_dirs = snapshot_task_run_dirs(normalized_task_id)
            cmd = [
                sys.executable,
                "-m",
                "mllm_base_agent.dual_agent.ai2thor.main",
                "--config",
                str(actual_config_path),
                "--task",
                normalized_task_id,
                "--output-dir",
                str(benchmark_task_output_dir),
            ]
            if args.agent1:
                cmd.extend(["--agent1", args.agent1])
            if args.agent2:
                cmd.extend(["--agent2", args.agent2])
            if args.max_steps:
                cmd.extend(["--max-steps", str(args.max_steps)])
            if getattr(args, "recursion_limit", None):
                cmd.extend(["--recursion-limit", str(args.recursion_limit)])
            if args.switch_interval:
                cmd.extend(["--switch-interval", str(args.switch_interval)])

            execution_start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=None,
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

            produced_run_dir = None
            result_json = find_result_json(str(benchmark_task_output_dir))
            if result_json is None:
                produced_run_dir = find_new_task_run_dir(
                    normalized_task_id,
                    before_dirs,
                    execution_start_time,
                )
                if produced_run_dir:
                    copied_dir = copytree_no_delete(produced_run_dir, benchmark_task_output_dir)
                    benchmark_task_output_dir = copied_dir

            task_log_parts.append("=== Output dir ===\n")
            if produced_run_dir:
                task_log_parts.append(f"Produced run dir: {produced_run_dir}\n")
                task_log_parts.append(f"Copied to: {benchmark_task_output_dir}\n\n")
            else:
                task_log_parts.append(f"Task output dir: {benchmark_task_output_dir}\n\n")

            task_log_parts.append("=== Run info ===\n")
            task_log_parts.append(f"Command: {' '.join(cmd)}\n")
            task_log_parts.append(f"Exit code: {result.returncode}\n")
            task_log_parts.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_parts.append(f"{'=' * 80}\n\n")

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
                    task_id,
                    str(benchmark_task_output_dir),
                    str(outputs_completed_path),
                )
                if csv_path:
                    csv_extra_fields = build_csv_extra_fields(
                        task_metadata,
                        actual_actions_info,
                        token_stats,
                        failure_reason_detail,
                    )
                    update_csv_task_record(
                        csv_path,
                        task_id,
                        status="true",
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                print(f"  ✅ {task_id} success")
            else:
                failure_reason = determine_failure_reason(task_log, result_json)
                csv_status = decide_csv_status_from_result(result_info, failure_reason)
                failure_reason_detail = result_info.get("fail_reason") or failure_reason
                csv_extra_fields = build_csv_extra_fields(
                    task_metadata,
                    actual_actions_info,
                    token_stats,
                    failure_reason_detail,
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
                            task_id,
                            str(benchmark_task_output_dir),
                            str(outputs_completed_path),
                        )

                    if csv_path:
                        update_csv_task_record(
                            csv_path,
                            task_id,
                            status=csv_status,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                else:
                    task_status = "failed_external"
                    failure_reason_detail = "No result JSON produced"
                    err_blob = f"{stderr_text}\n{stdout_text}".lower()
                    if "graphrecursionerror" in err_blob or "recursion limit" in err_blob:
                        failure_reason_detail = (
                            "GraphRecursionError: graph recursion_limit exceeded "
                            "(no dual_episode JSON saved; pull latest mllm_base_agent/dual_agent/ai2thor/main.py)"
                        )
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir),
                        task_id,
                        result.returncode,
                        stdout_text,
                        stderr_text,
                        error_text=failure_reason_detail,
                    )
                    if csv_path:
                        update_csv_task_record(
                            csv_path,
                            task_id,
                            status=None,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                    print(f"  ⚠️  {task_id} no result JSON -> null")

        except KeyboardInterrupt:
            task_status = "interrupted"
            task_log = "".join(task_log_parts)
            save_task_log(task_id, task_log, task_logs_dir, 1, "interrupted")
            raise
        except Exception as e:
            error_msg = str(e)
            task_log_parts.append("=== Exception ===\n")
            task_log_parts.append(f"Error: {error_msg}\n")
            task_log_parts.append(f"{'=' * 80}\n\n")
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
            csv_extra_fields = build_csv_extra_fields(
                task_metadata,
                actual_actions_info,
                token_stats,
                failure_reason_detail,
            )

            if failure_reason in ("parse_error", "action_error", "model_error"):
                task_status = "failed_model"
                if csv_status == "false":
                    copied_to_completed = copy_to_outputs_completed(
                        task_id,
                        str(benchmark_task_output_dir),
                        str(outputs_completed_path),
                    )
            else:
                task_status = "failed_external"
                if not result_json:
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir),
                        task_id,
                        None,
                        "",
                        "",
                        error_text=error_msg,
                    )

            if csv_path:
                update_csv_task_record(
                    csv_path,
                    task_id,
                    status=csv_status,
                    extra_fields=csv_extra_fields,
                    lock=csv_lock,
                )
            print(f"  ❌ {task_id} exception: {error_msg}")
        finally:
            with INFLIGHT_TASKS_LOCK:
                INFLIGHT_TASKS.discard(normalized_task_id)

        task_duration = time.time() - task_start_time
        return {
            "task_id": task_id,
            "status": task_status,
            "attempts": 1,
            "duration": task_duration,
            "success": task_success,
            "golden_actions_count": task_metadata.get("golden_action_count"),
            "actual_actions_count": actual_actions_info.get("actual_action_count"),
            "golden_action": task_metadata.get("golden_action_text"),
            "actual_actions": actual_actions_info.get("actual_action_text"),
            "instruction": task_metadata.get("instruction"),
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "failure_reason": failure_reason_detail,
            "task_result": result_info["task_result"],
            "agent_1_steps": result_info.get("agent_1_steps", 0),
            "agent_2_steps": result_info.get("agent_2_steps", 0),
            "communication_events": result_info.get("communication_events", 0),
            "turn_count": result_info.get("turn_count", 0),
            "copied_to_completed": copied_to_completed,
        }

    task_records: List[Dict[str, Any]] = []
    successful = 0
    failed_model = 0
    failed_external = 0
    copied_to_completed_count = 0
    exit_code = 0

    try:
        if args.sequential:
            if HAS_TQDM:
                task_iterator = tqdm(task_ids, desc="Tasks", unit="task", ncols=100)
            else:
                task_iterator = task_ids

            for idx, task_id in enumerate(task_iterator, 1):
                if not HAS_TQDM:
                    print(f"\n{'=' * 80}")
                    print(f"📋 Task {idx}/{len(task_ids)}: {task_id}")
                    print(f"{'=' * 80}")
                result = execute_task(task_id)
                task_records.append(result)
                if result["success"]:
                    successful += 1
                elif result["status"] == "failed_model":
                    failed_model += 1
                else:
                    failed_external += 1
                if result.get("copied_to_completed"):
                    copied_to_completed_count += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {
                    executor.submit(execute_task, task_id): task_id for task_id in task_ids
                }
                if HAS_TQDM:
                    task_iterator = tqdm(
                        as_completed(future_to_task),
                        total=len(task_ids),
                        desc="Tasks",
                        unit="task",
                        ncols=100,
                    )
                else:
                    task_iterator = as_completed(future_to_task)

                for future in task_iterator:
                    result = future.result()
                    task_records.append(result)
                    if result["success"]:
                        successful += 1
                    elif result["status"] == "failed_model":
                        failed_model += 1
                    else:
                        failed_external += 1
                    if result.get("copied_to_completed"):
                        copied_to_completed_count += 1
                    if HAS_TQDM:
                        task_iterator.set_postfix(
                            {
                                "ok": successful,
                                "model": failed_model,
                                "external": failed_external,
                                "copied": copied_to_completed_count,
                            }
                        )
    except KeyboardInterrupt:
        print("\n⚠️ User interrupt")
        exit_code = 1
    else:
        exit_code = 0 if (failed_model + failed_external) == 0 else 1
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                print("✓ Temp config removed")
            except Exception as e:
                print(f"⚠️ Failed to remove temp config: {e}")

    total_duration = sum(record["duration"] for record in task_records)
    avg_duration = total_duration / len(task_records) if task_records else 0
    aggregate_agent_stats = {
        "agent_1_steps": sum(record.get("agent_1_steps", 0) for record in task_records),
        "agent_2_steps": sum(record.get("agent_2_steps", 0) for record in task_records),
        "communication_events": sum(record.get("communication_events", 0) for record in task_records),
        "turn_count": sum(record.get("turn_count", 0) for record in task_records),
    }

    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write("Dual benchmark run summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if csv_path:
            f.write(f"CSV: {csv_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Actual config: {actual_config_path}\n")
        if args.agent1:
            f.write(f"Agent 1 config: {args.agent1}\n")
        if args.agent2:
            f.write(f"Agent 2 config: {args.agent2}\n")
        f.write(f"Output dir: {benchmark_output_dir}\n")
        f.write(f"Mode: {'sequential' if args.sequential else f'parallel (workers: {args.workers})'}\n")
        f.write(f"Collaboration mode: {args.collaboration_mode}\n")
        if args.max_steps:
            f.write(f"Max steps: {args.max_steps}\n")
        if args.switch_interval:
            f.write(f"Switch interval: {args.switch_interval}\n")
        if args.headless:
            f.write("Headless: enabled\n")
        f.write(f"\n{'=' * 80}\n")
        f.write("Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Total tasks: {len(task_ids)}\n")
        if task_ids:
            f.write(f"Success: {successful} ({successful / len(task_ids) * 100:.1f}%)\n")
            f.write(f"Model failure: {failed_model} ({failed_model / len(task_ids) * 100:.1f}%)\n")
            f.write(f"External failure: {failed_external} ({failed_external / len(task_ids) * 100:.1f}%)\n")
        f.write(f"Copied to {args.outputs_completed_dir}: {copied_to_completed_count}\n")
        f.write(f"Total time: {total_duration:.2f}s ({total_duration / 60:.2f} min)\n")
        f.write(f"Avg time: {avg_duration:.2f}s\n")
        f.write("\nDual-agent aggregate stat.")
        f.write(f"  Agent 1 steps: {aggregate_agent_stats['agent_1_steps']}\n")
        f.write(f"  Agent 2 steps: {aggregate_agent_stats['agent_2_steps']}\n")
        f.write(f"  Communication events: {aggregate_agent_stats['communication_events']}\n")
        f.write(f"  Turn count: {aggregate_agent_stats['turn_count']}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write("Task details\n")
        f.write(f"{'=' * 80}\n\n")
        for i, record in enumerate(task_records, 1):
            status_icon = "✅" if record["success"] else "❌"
            fail_reason = record.get("failure_reason") or "N/A"
            token_str = str(record.get("total_tokens")) if record.get("total_tokens") is not None else "N/A"
            actual_action_str = (
                str(record.get("actual_actions_count"))
                if record.get("actual_actions_count") is not None
                else "N/A"
            )
            golden_action_str = (
                str(record.get("golden_actions_count"))
                if record.get("golden_actions_count") is not None
                else "N/A"
            )
            f.write(
                f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                f"status: {record['status']:15s} | "
                f"duration: {record['duration']:8.2f}s | "
                f"golden_actions: {golden_action_str:>4s} | "
                f"actual_actions: {actual_action_str:>4s} | "
                f"tokens: {token_str:>8s}"
            )
            if record.get("failure_reason") and not record["success"]:
                f.write(f" | reason: {fail_reason}")
            f.write("\n")

    summary_json = {
        "timestamp": timestamp,
        "csv": str(csv_path) if csv_path else None,
        "config": str(config_path),
        "actual_config": str(actual_config_path),
        "agent1_config": args.agent1,
        "agent2_config": args.agent2,
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
        "copied_to_completed": copied_to_completed_count,
        "duration_seconds": total_duration,
        "avg_duration_seconds": avg_duration,
        "agent_statistics": aggregate_agent_stats,
        "task_records": task_records,
    }
    summary_json_path = benchmark_output_dir / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print("🎉 Dual benchmark complete")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Success: {successful}")
    print(f"Model failure: {failed_model}")
    print(f"External failure: {failed_external}")
    print(f"Copied to {args.outputs_completed_dir}: {copied_to_completed_count}")
    print(f"Agent 1 steps: {aggregate_agent_stats['agent_1_steps']}")
    print(f"Agent 2 steps: {aggregate_agent_stats['agent_2_steps']}")
    print(f"Communication events: {aggregate_agent_stats['communication_events']}")
    print(f"Turn count: {aggregate_agent_stats['turn_count']}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Task logs: {task_logs_dir}")
    print(f"Summary log: {summary_log_path}")
    print(f"Summary json: {summary_json_path}")

    if csv_path:
        csv_stats = count_csv_status(csv_path)
        print(f"\n{'=' * 80}")
        print("📊 CSV status")
        print(f"{'=' * 80}")
        total_csv = csv_stats["total"]
        if total_csv > 0:
            true_count = csv_stats["true"]
            false_count = csv_stats["false"]
            null_count = csv_stats["null"]
            print(f"Total: {total_csv}")
            print(f"true: {true_count} ({true_count / total_csv * 100:.1f}%)")
            print(f"false: {false_count} ({false_count / total_csv * 100:.1f}%)")
            print(f"null: {null_count} ({null_count / total_csv * 100:.1f}%)")

    print(f"{'=' * 80}\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
