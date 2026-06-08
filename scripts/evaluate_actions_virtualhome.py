"""
VirtualHome         
============================
           VirtualHome      ，       
   scripts/ai2thor/evaluate_action_sequence.py    ，   VirtualHome    

VirtualHome   AI2-THOR      ：
  -     ：WalkForward TurnLeft Grab(apple) PutBack(apple, table)  
  -   ：   0-49（    ）
  -     ：   URL:Port        Unity   （  CloudRendering）
  -     ：  （graph）   metadata，      wrapper     

#     

##   1：    task.json      golden_actions（  ）
python scripts/virtualhome/evaluate_action_sequence.py --task virtualhome001
#      tasks/virtualhome001/task.json     golden_actions

##   2：     （config.yaml task_presets）  
python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge

##   3：             
python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge --actions "TurnLeft,WalkForward,Open(fridge),DONE"

##   4：         
python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge --action-file actions.txt

##   5：    （    ）
python scripts/virtualhome/evaluate_action_sequence.py --step-by-step --task open_fridge

VirtualHome     ：
  -   （   ）: TurnLeft, TurnRight, WalkForward
  -      : Grab(apple), Open(fridge), Close(microwave), SwitchOn(tv)
  -      : PutBack(apple, table), PutIn(glass, microwave)
  -     : DONE, FAIL
"""

import os
import sys
import json
import argparse
import platform
import shlex
import subprocess
import shutil
import socket
import importlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

#           
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
REPO_ROOT = str(Path(PROJECT_ROOT).parent)
for import_root in (REPO_ROOT, PROJECT_ROOT):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from mllm_base_agent.console import configure_utf8_stdio

configure_utf8_stdio()

from config import load_config
from mllm_base_agent.environments.virtualhome.backend_utils import (
    DEFAULT_BACKEND_ARGS,
    DEFAULT_BACKEND_EXE,
    DEFAULT_STARTUP_TIMEOUT,
    build_backend_command,
    resolve_backend_args,
    resolve_backend_exe,
    resolve_backend_host,
    resolve_backend_port,
    resolve_backend_startup_timeout,
)

# ============================================================================
# ★       （    ，        ）
# ============================================================================

# VirtualHome        （Windows   ）
_DEFAULT_BACKEND_EXE = DEFAULT_BACKEND_EXE

#           
_DEFAULT_BACKEND_ARGS = DEFAULT_BACKEND_ARGS

#               （ ）
_DEFAULT_STARTUP_TIMEOUT = DEFAULT_STARTUP_TIMEOUT

# ============================================================================
# Backend     /    （    run_all_tasks_evaluate_manual.py）
# ============================================================================


def _is_port_open(host: str, port: int, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    """              ，       """
    if timeout_sec <= 0:
        return True
    start = time.time()
    while time.time() - start < timeout_sec:
        if _is_port_open(host, port):
            return True
        time.sleep(1.0)
    return False


def _launch_backend(
    backend_exe: str,
    backend_args_str: str,
    port: int,
):
    """Launch the VirtualHome backend and return the process."""
    cmd, exe_path = build_backend_command(
        backend_exe,
        backend_args_str,
        port=port,
    )
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(exe_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    print(f"✓ Started VirtualHome backend (PID={proc.pid}): {exe_path.name}")
    return proc


def _terminate_backend(proc, timeout_sec: int = 20) -> None:
    """         """
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass

#    VirtualHome        /     
from envs.virtualhome.utils import (
    parse_vh_action_string,
    format_vh_action_dict,
    coerce_init_rotation_quat,
    quat_to_yaw_deg,
    quantize_yaw_degrees,
)

#    graph.py         
from core.agent.graph import perform_final_evaluation, execute_action


DIRECT_DONE_FAILURE_REASON = "Rejected direct DONE before any task action"


# ============================================================================
# VirtualHome     （  envs/virtualhome/utils.py     ）
# ============================================================================

VH_NAVIGATION_WITH_OBJECT = set()
VH_NAVIGATION_NO_OBJECT = {
    "TurnLeft",
    "TurnRight",
    "WalkForward",
    "LookUp",
    "LookDown",
    "StandUp",
}
VH_INTERACTION_ONE_OBJECT = {
    "Grab",
    "Open",
    "Close",
    "SwitchOn",
    "SwitchOff",
    "Drink",
    "Sit",
    "LookAt",
    "Touch",
}
VH_INTERACTION_TWO_OBJECTS = {"PutBack", "PutIn"}


def parse_action_sequence(action_str: str) -> List[Dict[str, Any]]:
    """
             

       envs/virtualhome/utils.py    parse_vh_action_string()        

        ：
    - TurnLeft,WalkForward,Open(fridge)
    - TurnLeft,WalkForward,TurnRight
    - Grab(apple),PutBack(apple, table)
    - DONE, FAIL

    Args:
        action_str:        ，     
                      ：                ，
                               

    Returns:
                
    """
    actions = []

    #     ：           
    #    "PutBack(apple, table),WalkForward" → ["PutBack(apple, table)", "WalkForward"]
    tokens = _smart_split(action_str)

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        try:
            action_dict = parse_vh_action_string(token)
            actions.append(action_dict)
        except ValueError as e:
            print(f"⚠️       '{token}'   : {e}")
            continue

    return actions


def _smart_split(s: str) -> List[str]:
    """
           ：            

       VirtualHome          ，  ：
    "PutBack(apple, table),WalkForward" → ["PutBack(apple, table)", "WalkForward"]

    Args:
        s:        

    Returns:
             token   
    """
    tokens = []
    depth = 0
    current = []

    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        tokens.append("".join(current))

    return tokens


def _normalize_structured_action(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a structured VH action dict against the prompt action space."""
    if not isinstance(action_dict, dict):
        raise ValueError(f"Expected dict action, got {type(action_dict).__name__}")

    formatted = format_vh_action_dict(action_dict)
    normalized = parse_vh_action_string(formatted)

    for key in (
        "object_id",
        "object2_id",
        "turn_modifier",
        "turn_degrees",
        "granularity",
        "magnitude",
    ):
        if key not in action_dict:
            continue
        value = action_dict.get(key)
        if key in ("object_id", "object2_id") and value not in (None, ""):
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        normalized[key] = value

    return normalized


def _merge_recorded_user_actions(
    action_sequence: List[Dict[str, Any]],
    recorded_user_actions: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Inject interact-recorded prompt metadata / instance ids into the eval action list."""
    if not action_sequence or not recorded_user_actions:
        return action_sequence

    normalized_recorded: List[Dict[str, Any]] = []
    for idx, item in enumerate(recorded_user_actions, 1):
        try:
            normalized_recorded.append(_normalize_structured_action(item))
        except Exception as exc:
            print(f"⚠️      {idx}   recorded_user_actions: {exc}")

    if not normalized_recorded:
        return action_sequence

    def _action_signature(action: Dict[str, Any]) -> Tuple[Any, ...]:
        action_name = str(action.get("action_name", "")).strip()
        action_type = str(action.get("action_type", "")).strip()
        object_type = action.get("object_type")
        object2_type = action.get("object2_type")
        granularity = action.get("granularity")
        turn_modifier = action.get("turn_modifier")
        turn_degrees = action.get("turn_degrees")

        if action_name == "WalkForward" and not granularity and action.get("magnitude") is None:
            granularity = "small"

        if action_name in ("TurnLeft", "TurnRight"):
            if turn_modifier:
                turn_key = str(turn_modifier).lower()
            elif turn_degrees is not None:
                try:
                    turn_key = "small" if abs(float(turn_degrees) - 30.0) < 1e-6 else "normal"
                except (TypeError, ValueError):
                    turn_key = "normal"
            else:
                turn_key = "normal"
        else:
            turn_key = None

        return (
            action_type,
            action_name,
            object_type,
            object2_type,
            granularity,
            turn_key,
        )

    merged: List[Dict[str, Any]] = []
    merged_count = 0
    compare_len = min(len(action_sequence), len(normalized_recorded))

    for idx, action in enumerate(action_sequence):
        merged_action = dict(action)
        if idx < compare_len:
            recorded = normalized_recorded[idx]
            if _action_signature(merged_action) == _action_signature(recorded):
                for key in (
                    "object_id",
                    "object2_id",
                    "turn_modifier",
                    "turn_degrees",
                    "granularity",
                    "magnitude",
                ):
                    if key in recorded:
                        merged_action[key] = recorded[key]
                merged_count += 1
            else:
                print(
                    f"⚠️  recorded_user_actions entry {idx + 1} does not match the task action; "
                    f"using task action: task={format_vh_action_dict(merged_action)} | "
                    f"recorded={format_vh_action_dict(recorded)}"
                )
        merged.append(merged_action)

    if merged_count:
        print(f"✓    init.json    {merged_count}   interact       ")

    return merged


def load_discrete_init_data(
    init_file: str,
    yaw_step_degrees: float = 30.0,
) -> Dict[str, Any]:
    """Load init.json and normalize rotation to 30-degree discrete yaw.

    Returns a dict with:
      - init_action_strings
      - init_scene
      - init_char_position
      - init_char_rotation (discrete quaternion)
      - init_char_yaw_degrees (discrete yaw degrees)
      - init_camera_pitch
      - recorded_user_actions
      - error (optional)
    """
    result: Dict[str, Any] = {
        "init_action_strings": [],
        "init_scene": None,
        "init_char_position": None,
        "init_char_rotation": None,
        "init_char_yaw_degrees": None,
        "init_camera_pitch": None,
        "recorded_user_actions": None,
    }

    if not os.path.exists(init_file):
        return result

    try:
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)
    except Exception as e:
        result["error"] = str(e)
        return result

    if isinstance(init_data, list):
        result["init_action_strings"] = init_data
    elif isinstance(init_data, dict):
        result["init_action_strings"] = init_data.get("actions", [])
        result["init_scene"] = init_data.get("scene")
        result["init_char_position"] = init_data.get("character_position")
        result["init_char_rotation"] = init_data.get("character_rotation")
        result["init_char_yaw_degrees"] = init_data.get("character_yaw_degrees")
        if result["init_char_yaw_degrees"] is None:
            yaw_from_quat = quat_to_yaw_deg(result["init_char_rotation"])
            if yaw_from_quat is not None:
                result["init_char_yaw_degrees"] = quantize_yaw_degrees(
                    yaw_from_quat, yaw_step_degrees
                )
        result["init_char_rotation"] = coerce_init_rotation_quat(
            char_rotation=result["init_char_rotation"],
            char_yaw_degrees=result["init_char_yaw_degrees"],
            step_degrees=yaw_step_degrees,
        )
        result["init_camera_pitch"] = init_data.get("camera_pitch")
        result["recorded_user_actions"] = init_data.get("recorded_user_actions")
    else:
        result["error"] = "init.json must be a list or dict"
        return result

    #      Done
    init_action_strings = result.get("init_action_strings") or []
    if init_action_strings and str(init_action_strings[-1]).strip().upper() == "DONE":
        result["init_action_strings"] = init_action_strings[:-1]

    return result


def extract_golden_actions_from_task(
    task_config: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
             golden_actions

          ：
    1.        （  ）：
       "golden_actions": {
         "steps": 3,
         "actions": ["TurnLeft", "WalkForward", "Open(fridge)", "Done"]
       }

    2.          ：
       "golden_actions": {
         "steps": 3,
         "actions": [
           {"action_type": "navigation", "action_name": "WalkForward"},
           ...
         ]
       }

    3.          （   ）：
       "golden_actions": "TurnLeft,WalkForward,Open(fridge),DONE"

    Args:
        task_config:       

    Returns:
              ，         None
    """
    golden_actions = task_config.get("golden_actions")

    if not golden_actions:
        return None

    #   1&2:     （   steps   actions）
    if isinstance(golden_actions, dict):
        actions = golden_actions.get("actions")
        if isinstance(actions, list) and len(actions) > 0:
            #          
            if all(isinstance(a, dict) and "action_name" in a for a in actions):
                normalized_actions = []
                for idx, action_dict in enumerate(actions, 1):
                    try:
                        normalized_actions.append(_normalize_structured_action(action_dict))
                    except ValueError as e:
                        print(f"⚠️    {idx}         : {e}")
                print(
                    f"✓          golden_actions (         , {len(actions)}    )"
                )
                return normalized_actions if normalized_actions else None

            #        （  ）
            elif all(isinstance(a, str) for a in actions):
                print(
                    f"✓          golden_actions (       , {len(actions)}    )"
                )
                parsed_actions = []
                for action_str in actions:
                    try:
                        if action_str.strip().upper() == "DONE":
                            parsed_actions.append(
                                {
                                    "action_type": "task_completion",
                                    "action_name": "DONE",
                                }
                            )
                            break
                        action_dict = parse_vh_action_string(action_str)
                        parsed_actions.append(action_dict)
                    except ValueError as e:
                        print(f"⚠️       '{action_str}'   : {e}")
                        continue
                return parsed_actions if parsed_actions else None

    #   3:          
    elif isinstance(golden_actions, str):
        print("✓          golden_actions (         )")
        return parse_action_sequence(golden_actions)

    return None


def convert_action_sequence_to_string_array(
    action_sequence: List[Dict[str, Any]],
) -> List[str]:
    """
         （    ）          

          task.json   golden_actions.actions   
           "Done"   

    VirtualHome     ：
      WalkForward                    —      
      TurnLeft                —      
      Grab(apple)             —      
      PutBack(apple, table)   —      

    Args:
        action_sequence:     

    Returns:
             
    """
    action_strings = []

    for action in action_sequence:
        action_name = str(action.get("action_name", "")).upper()
        if action_name in ("DONE", "FAIL"):
            continue
        action_strings.append(format_vh_action_dict(action))

    action_strings.append("Done")
    return action_strings


def determine_task_level(action_sequence: List[Dict[str, Any]]) -> str:
    """
                  

    - Level1 Navigation:     navigation   
    - Level2 Interaction:     interaction   
    - Level3 Hybrid:      navigation   interaction   

    Args:
        action_sequence:     

    Returns:
               
    """
    has_navigation = False
    has_interaction = False

    for action in action_sequence:
        action_type = action.get("action_type", "")
        if action_type == "navigation":
            has_navigation = True
        elif action_type == "interaction":
            has_interaction = True

    if has_navigation and has_interaction:
        return "Level3 Hybrid"
    elif has_interaction:
        return "Level2 Interaction"
    elif has_navigation:
        return "Level1 Navigation"
    else:
        return "Unknown"


def generate_csv_row(
    task_config: Dict[str, Any], action_sequence: List[Dict[str, Any]], task_id: str
) -> Dict[str, str]:
    """
       CSV    

                  （ 16 ），  AI2-THOR        

    Args:
        task_config:     
        action_sequence:     
        task_id:    ID

    Returns:
        CSV      
    """
    action_strings = convert_action_sequence_to_string_array(action_sequence)
    golden_actions_str = json.dumps(action_strings, ensure_ascii=False)
    level = determine_task_level(action_sequence)
    steps = len(action_strings)

    row = {
        "Task ID": task_id,
        "Task Name": task_config.get("task_name", "") or task_config.get("name", ""),
        "Instruction": task_config.get("instruction", "")
        or task_config.get("description", ""),
        "Golden Action": golden_actions_str,
        "Step Number": str(steps),
        "Evaluation": "",
        "Category": "Daily Household",
        "Level": level,
        "Evaluation Type": "Conditional",
        "Plan": "",
        "Env": "virtualhome",
        "Comment": "",
        "Annotation": "",
        "Anotator": "",
        "Check": "",
        "Checker": "",
    }

    return row


def copy_to_clipboard(text: str):
    """           """
    system = platform.system()

    if system == "Darwin":
        try:
            subprocess.run("pbcopy", text=True, input=text, check=True)
            print("✔ Copied to clipboard (macOS)")
            return
        except Exception:
            pass
    elif system == "Linux":
        for cmd in [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]:
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, text=True, input=text, check=True)
                    print(f"✔ Copied to clipboard (Linux + {cmd[0]})")
                    return
                except Exception:
                    pass
        print("⚠️  Linux clipboard copy needs xclip or xsel")
    elif system == "Windows":
        try:
            subprocess.run("clip", text=True, input=text, check=True)
            print("✔ Copied to clipboard (Windows)")
            return
        except Exception:
            pass


def _port_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    """Check TCP reachability for Unity endpoint."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _try_import_virtualhome_with_fallback() -> Tuple[bool, str]:
    """Import VirtualHome package; fallback to common local clone paths."""
    try:
        mod = importlib.import_module(
            "virtualhome.simulation.unity_simulator.comm_unity"
        )
        return True, getattr(mod, "__file__", "(unknown)")
    except Exception:
        pass

    #    pip virtualhome   __init__.py     ：
    #           sys.path     simulation.*
    try:
        spec = importlib.util.find_spec("virtualhome")
    except Exception:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        package_root = Path(list(spec.submodule_search_locations)[0])
        if (package_root / "simulation").exists() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        try:
            mod = importlib.import_module("simulation.unity_simulator.comm_unity")
            return True, getattr(mod, "__file__", f"{package_root} (simulation.*)")
        except Exception:
            pass

    candidates = [
        Path(PROJECT_ROOT).parent / "virtualhome",
        Path.cwd().parent / "virtualhome",
    ]
    for p in candidates:
        if (p / "setup.py").exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            try:
                mod = importlib.import_module(
                    "virtualhome.simulation.unity_simulator.comm_unity"
                )
                return True, getattr(mod, "__file__", f"{p} (path-added)")
            except Exception:
                continue
    return False, ""


def print_runtime_diagnostics(config_obj):
    """Print environment diagnostics for cross-machine reproducibility."""
    host = str(config_obj.get("env.url", "127.0.0.1"))
    port_raw = str(config_obj.get("env.port", "8080"))
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080

    ok_import, module_path = _try_import_virtualhome_with_fallback()

    print("\n" + "=" * 70)
    print("VirtualHome Runtime Diagnostics")
    print("=" * 70)
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Unity endpoint: {host}:{port}")
    print(f"TCP reachable: {'YES' if _port_reachable(host, port) else 'NO'}")
    print(f"virtualhome import: {'OK' if ok_import else 'FAILED'}")
    if ok_import:
        print(f"virtualhome module: {module_path}")
    else:
        print("hint: install by either")
        print("  1) pip install virtualhome==2.3.0")
        print("  2) git clone virtualhome && pip install -e .")
        print("Both are valid; editable install is NOT mandatory.")
    print("=" * 70 + "\n")


def _configure_env_from_task(env, task_config: Dict[str, Any]):
    """  task_config     success_conditions     env 

       success_predicate   wrapper   reset/step   ，
         target_object_types   target_description 
    """
    target_types = task_config.get("target_object_types", [])
    description = task_config.get("target_description", "") or task_config.get(
        "instruction", ""
    )

    conditions = task_config.get("success_conditions", [])
    condition = task_config.get("success_condition", {})

    #    success_predicate
    predicate = None

    if conditions:
        #      object_state     （wrapper      predicate）
        for cond in conditions:
            ctype = cond.get("type", "object_state")
            if ctype == "object_state":
                field = cond.get("state") or cond.get("field", "isOpen")
                value = cond.get("value", True)
                predicate = lambda obj, f=field, v=value: obj.get(f, False) == v
                break
            elif ctype == "object_in_receptacle":
                rec_type = cond.get("receptacle_type", "")
                exp = cond.get("value", True)

                def _check(obj, rt=rec_type, e=exp):
                    parents = obj.get("parentReceptacles", [])
                    in_rec = any(p == rt or p.startswith(rt) for p in parents)
                    return in_rec == e

                predicate = _check
                break
            elif ctype == "object_in_hand":
                predicate = lambda obj: obj.get("isPickedUp", False)
                break
    elif condition:
        ctype = condition.get("type", "object_state")
        if ctype == "object_state":
            field = condition.get("field", "isOpen")
            value = condition.get("value", True)
            predicate = lambda obj, f=field, v=value: obj.get(f, False) == v

    if predicate is None:
        predicate = lambda obj: False

    env.configure_task(
        target_types,
        predicate,
        description,
        success_condition=condition,
        success_conditions=conditions,
        success_logic=task_config.get("success_logic", "AND"),
    )


def _has_non_completion_action(action_sequence: List[Dict[str, Any]]) -> bool:
    return any(
        action.get("action_type") != "task_completion"
        and str(action.get("action_name", "")).upper() not in {"DONE", "FAIL"}
        for action in action_sequence or []
    )


def _condition_label(condition: Dict[str, Any]) -> str:
    ctype = condition.get("type", "object_state")
    obj = condition.get("object_type", "*")
    if ctype == "object_state":
        field = condition.get("state") or condition.get("field", "isOpen")
        return f"{ctype}:{obj}:{field}={condition.get('value', True)}"
    if ctype == "object_in_receptacle":
        return (
            f"{ctype}:{obj}->{condition.get('receptacle_type')}="
            f"{condition.get('value', True)}"
        )
    if ctype == "object_in_hand":
        return f"{ctype}:{obj}={condition.get('value', True)}"
    return f"{ctype}:{obj}"


def _evaluate_condition_diagnostics(env, task_config: Dict[str, Any], observation) -> Dict[str, Any]:
    diagnostics = {
        "success_logic": task_config.get("success_logic", "AND"),
        "conditions": [],
        "score": None,
        "success": None,
        "error": None,
    }
    if not observation or not getattr(observation, "metadata", None):
        diagnostics["error"] = "No observation metadata"
        return diagnostics

    conditions = task_config.get("success_conditions")
    if isinstance(conditions, dict):
        conditions = [conditions]
    elif not isinstance(conditions, list) or not conditions:
        condition = task_config.get("success_condition")
        conditions = [condition] if isinstance(condition, dict) else []
    target_types = task_config.get("target_object_types") or []
    normalized_conditions = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        normalized = dict(condition)
        if (
            normalized.get("type", "object_state") == "object_state"
            and not normalized.get("object_type")
            and len(target_types) == 1
        ):
            normalized["object_type"] = target_types[0]
        normalized_conditions.append(normalized)
    conditions = normalized_conditions

    try:
        from evaluation.procthor.base import MultiConditionEvaluator, create_evaluator_from_config

        if conditions:
            for condition in conditions:
                if not isinstance(condition, dict):
                    continue
                evaluator = MultiConditionEvaluator([condition], logic="AND")
                score = evaluator.evaluate(env, observation.metadata)
                diagnostics["conditions"].append(
                    {
                        "label": _condition_label(condition),
                        "condition": condition,
                        "score": score,
                        "passed": score >= 1.0,
                    }
                )
        evaluator = create_evaluator_from_config(task_config)
        final_score = evaluator.evaluate(env, observation.metadata)
        diagnostics["score"] = final_score
        diagnostics["success"] = final_score >= 1.0
    except Exception as exc:
        diagnostics["error"] = str(exc)
    return diagnostics


def _write_evaluation_diagnostics(output_dir: str, diagnostics: Dict[str, Any]) -> str:
    path = os.path.join(output_dir, "evaluation_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(f"🧪 Evaluation diagnostics saved: {path}")
    return path


def _summarize_condition_diagnostics(condition_diag: Optional[Dict[str, Any]]) -> str:
    if not isinstance(condition_diag, dict):
        return ""
    parts = []
    for condition in condition_diag.get("conditions", []) or []:
        label = condition.get("label", "condition")
        status = "PASS" if condition.get("passed") else "FAIL"
        parts.append(f"{label}:{status}")
    if parts:
        return "; ".join(parts)
    if condition_diag.get("error"):
        return f"error:{condition_diag['error']}"
    return ""


def _first_failure_step(executed_actions: List[Dict[str, Any]]) -> Optional[int]:
    for step in executed_actions or []:
        error_message = step.get("error_message")
        last_action_success = step.get("last_action_success")
        success = step.get("success")
        if error_message or last_action_success is False or success is False:
            return step.get("step")
    return None


def execute_action_sequence(
    env,  # VirtualHomeEnvWrapper
    action_sequence: List[Dict[str, Any]],
    task_config: Dict[str, Any],
    output_dir: str,
    scene: Optional[int] = None,
    init_action_strings: List[str] = None,
    step_by_step: bool = False,
    char_position=None,
    char_rotation=None,
    char_yaw_degrees=None,
    camera_pitch=None,
    recorded_user_actions: Optional[List[Dict[str, Any]]] = None,
    backend_mode: str = "isolated",
) -> Dict[str, Any]:
    """
      VirtualHome               

       core/agent/graph.py    execute_action()   perform_final_evaluation() 
    VirtualHomeEnvWrapper     AI2-THOR     metadata，         

    Args:
        env: VirtualHomeEnvWrapper   
        action_sequence:     
        task_config:     
        output_dir:     
        scene:     （    /  ；reset      Unity   ）
        init_action_strings:           （  ）
        step_by_step:           
        char_position:        [x, y, z]（       ）
        char_rotation:           [x, y, z, w]（     ）
        char_yaw_degrees:         （ ，  ）
        camera_pitch:    （ ），    ，    

    Returns:
              
    """
    print(f"\n{'=' * 60}")
    print("VirtualHome action sequence")
    print(f"{'=' * 60}")

    result = {
        "task_name": task_config.get("name", "")
        or task_config.get("task_name", "custom_sequence"),
        "action_sequence": action_sequence,
        "executed_actions": [],
        "success": False,
        "evaluation_score": 0.0,
        "step_count": 0,
        "error_messages": [],
        "final_state": None,
        "init_action_count": 0,
        "failure_reason": None,
    }

    diagnostics: Dict[str, Any] = {
        "backend_mode": backend_mode,
        "requested_scene": scene,
        "initial_success": None,
        "initial_conditions": None,
        "direct_done_rejected": False,
        "direct_done_reject_reason": None,
        "executed_actions": [],
        "final_conditions": None,
        "failure_reason": None,
        "failure_step": None,
    }

    #       runner            ；             init    
    description = task_config.get("description", "Execute action sequence")
    resolved_char_rotation = coerce_init_rotation_quat(
        char_rotation=char_rotation,
        char_yaw_degrees=char_yaw_degrees,
        step_degrees=30.0,
    )

    observation = env.reset(
        description,
        scene=scene,
        char_position=char_position,
        char_rotation=resolved_char_rotation,
        char_yaw_degrees=char_yaw_degrees,
        camera_pitch=camera_pitch,
    )

    #    ：    （  ）     
    if char_position:
        print(f"\n{'=' * 60}")
        print("✓ Loaded initial pose from init.json")
        print(f"Initial position: {char_position}")
        if char_yaw_degrees is not None:
            print(f"Initial yaw: {float(char_yaw_degrees):.1f}°")
        elif resolved_char_rotation:
            import math

            qx, qy, qz, qw = (
                resolved_char_rotation[0],
                resolved_char_rotation[1],
                resolved_char_rotation[2],
                resolved_char_rotation[3],
            )
            yaw = math.degrees(
                math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
            )
            print(f"Initial yaw: {yaw:.1f}°")
        if camera_pitch is not None:
            print(f"Initial camera pitch: {camera_pitch}°")
        print(f"{'=' * 60}")

    elif init_action_strings:
        print(f"\n{'=' * 60}")
        print(f"📁 Executing init actions ({len(init_action_strings)} actions)")
        print(f"{'=' * 60}")

        for i, action_str in enumerate(init_action_strings, 1):
            action_str = action_str.strip()
            if not action_str or action_str.upper() == "DONE":
                break

            print(f"  {i}. Init action: {action_str}")

            try:
                action_dict = parse_vh_action_string(action_str)
                observation, error_message = env.step_with_action_dict(action_dict)
                result["init_action_count"] += 1

                if error_message:
                    print(f"     ⚠️  {error_message}")
                else:
                    print("     ✓   ")

            except Exception as e:
                print(f"     ❌ Init action parse/execution failed: {e}")
                continue

        print(f"\n✓ Init actions complete ({result['init_action_count']} executed)\n")

    initial_condition_diag = _evaluate_condition_diagnostics(env, task_config, observation)
    diagnostics["initial_conditions"] = initial_condition_diag
    diagnostics["initial_success"] = initial_condition_diag.get("success")

    # -- Execute golden action sequence via the canonical wrapper method ------
    # Golden replay intentionally uses the same strict interaction checks as
    # agent runtime: targets must be visible and CLOSE before interaction.
    # Recorded object IDs from init.json are still injected for instance
    # consistency, but they do not bypass reachability/visibility checks.

    executed_details: List[Dict[str, Any]] = []

    def _record_step(step_i, action_dict, obs, error_msg):
        if obs and hasattr(obs, "image_path") and obs.image_path:
            print(f"   snapshot: {obs.image_path}")
        success_flag = (
            obs.metadata.get("lastActionSuccess", True)
            if obs and obs.metadata else True
        )
        executed_details.append({
            "step": step_i,
            "action": action_dict,
            "success": success_flag,
            "error_message": error_msg,
            "reward": obs.reward if obs else 0,
            "done": obs.done if obs else False,
            "bound_instances": (
                dict(obs.metadata.get("bound_instances", {}))
                if obs and obs.metadata
                else {}
            ),
        })
        if obs and obs.done:
            print("info: done=True returned; continuing (final eval decides)")

    if not _has_non_completion_action(action_sequence):
        result["success"] = False
        result["evaluation_score"] = 0.0
        result["failure_reason"] = DIRECT_DONE_FAILURE_REASON
        result["failure_step"] = None
        result["error_messages"].append(DIRECT_DONE_FAILURE_REASON)
        diagnostics["direct_done_rejected"] = True
        diagnostics["direct_done_reject_reason"] = DIRECT_DONE_FAILURE_REASON
        diagnostics["failure_reason"] = DIRECT_DONE_FAILURE_REASON
        print(f"❌ {DIRECT_DONE_FAILURE_REASON}")
    elif step_by_step:
        merged = env.merge_recorded_user_actions(action_sequence, recorded_user_actions)
        for i, action_dict in enumerate(merged, 1):
            _aname = action_dict.get("action_name", "")
            _atype = action_dict.get("action_type", "")
            display = format_vh_action_dict(action_dict)
            if _atype == "task_completion":
                print(f"Task completion reached: {_aname}")
                executed_details.append({
                    "step": i, "action": action_dict, "success": True,
                    "error_message": None, "reward": 0, "done": True,
                    "is_task_completion": True,
                })
                result["step_count"] = i
                break
            print(f"\n[step-by-step] step {i}/{len(merged)}: {display}")
            input("Press Enter to execute...")
            obs, error_msg = env.step_with_action_dict(action_dict)
            print(f"  reward={obs.reward if obs else 0}, done={obs.done if obs else False}")
            if error_msg:
                print(f"  WARNING: {error_msg}")
                result["error_messages"].append(f"Step {i}: {error_msg}")
            _record_step(i, action_dict, obs, error_msg)
            result["step_count"] = i
            if obs:
                observation = obs
    else:
        exec_result = env.execute_golden_action_sequence(
            action_sequence,
            recorded_user_actions=recorded_user_actions,
            step_callback=lambda i, ad, obs, err: _record_step(i, ad, obs, err),
        )
        result["step_count"] = exec_result["steps"]
        result["error_messages"].extend(exec_result["errors"])
        diagnostics["executed_actions"] = exec_result.get("executed_actions", [])
        if exec_result["last_observation"] is not None:
            observation = exec_result["last_observation"]

    result["executed_actions"] = executed_details
    if not diagnostics["executed_actions"]:
        diagnostics["executed_actions"] = executed_details

    result["failure_step"] = _first_failure_step(diagnostics["executed_actions"])
    diagnostics["failure_step"] = result["failure_step"]

    #     
    if not diagnostics["direct_done_rejected"]:
        try:
            success, score = perform_final_evaluation(
                env=env, task_config=task_config, observation=observation
            )
            result["evaluation_score"] = score
            result["success"] = success
            if not success and not result.get("failure_reason"):
                result["failure_reason"] = "Final success conditions not met"
            print(f"\n📊 Final evaluation score: {score:.2f}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            result["error_messages"].append(f"Evaluation failed: {e}")
            result["failure_reason"] = f"Evaluation failed: {e}"

    final_condition_diag = _evaluate_condition_diagnostics(env, task_config, observation)
    diagnostics["final_conditions"] = final_condition_diag
    diagnostics["failure_reason"] = result.get("failure_reason")
    diagnostics["failure_step"] = result.get("failure_step")
    result["condition_diagnostics"] = final_condition_diag
    result["condition_summary"] = _summarize_condition_diagnostics(final_condition_diag)

    #       
    final_last_action_success = (
        observation.metadata.get("lastActionSuccess", True)
        if observation.metadata
        else True
    )
    result["final_state"] = {
        "done": observation.done,
        "reward": observation.reward,
        "text_state": observation.text_state,
        "last_action_success": final_last_action_success,
    }

    #    VirtualHome     （    ）
    if hasattr(env, "get_current_graph"):
        graph = env.get_current_graph()
        if graph:
            graph_file = os.path.join(output_dir, "final_graph.json")
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            print(f"📊 Graph saved: {graph_file}")

    #     
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"vh_action_result_{timestamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 Result saved: {result_file}")
    _write_evaluation_diagnostics(output_dir, diagnostics)
    return result


def main():
    """   """
    parser = argparse.ArgumentParser(
        description="Evaluate a VirtualHome action sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Task preset from config.yaml:
   python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge

2. Task file under tasks/:
   python scripts/virtualhome/evaluate_action_sequence.py --task virtualhome001

3. Custom actions:
   python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge --actions "TurnLeft,WalkForward,Open(fridge),DONE"

4. Action file:
   python scripts/virtualhome/evaluate_action_sequence.py --task open_fridge --action-file actions.txt

5. Step-by-step mode:
   python scripts/virtualhome/evaluate_action_sequence.py --step-by-step --task open_fridge

VirtualHome actions:
  - Movement: TurnLeft, TurnRight, WalkForward
  - Interaction: Grab(apple), Open(fridge), SwitchOn(tv)
  - Placement: PutBack(apple, table), PutIn(glass, microwave)
  - Completion: DONE, FAIL
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/virtualhome/config_close_gpt-5.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task ID under tasks/ or task preset name",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="Scene index (0-49); overrides config/task scene",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="Comma-separated action sequence",
    )
    parser.add_argument(
        "--action-file",
        type=str,
        default=None,
        help="File containing an action sequence",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Pause before each action and wait for Enter",
    )
    parser.add_argument(
        "--enable-visibility-check",
        action="store_true",
        help="Enable strict visibility checks before interaction",
    )
    # ──         （  run_all_tasks_evaluate_manual.py）────────────────
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Do not launch VirtualHome backend; use an already running backend",
    )
    parser.add_argument(
        "--backend-exe",
        default=None,
        help="VirtualHome.exe path (overrides config env.backend_exe)",
    )
    parser.add_argument(
        "--backend-args",
        default=None,
        help="Backend arguments (overrides config env.backend_args)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=None,
        help="Backend startup timeout in seconds (overrides config env.backend_startup_timeout)",
    )
    parser.add_argument(
        "--keep-backend",
        action="store_true",
        help="Leave launched backend running after evaluation",
    )
    args = parser.parse_args()
    _backend_cfg = load_config(args.config)
    args.backend_exe = resolve_backend_exe(_backend_cfg, args.backend_exe)
    args.backend_args = resolve_backend_args(_backend_cfg, args.backend_args)
    args.startup_timeout = resolve_backend_startup_timeout(
        _backend_cfg, args.startup_timeout
    )

    #     
    if not args.task and not args.actions and not args.action_file:
        print("❌ Missing input. Provide one of:")
        print(
            "   1. --task <task_id_or_preset> (task.json or task_presets golden_actions)"
        )
        print("   2. --actions <comma_separated_actions>")
        print("   3. --action-file <action_sequence_file>")
        sys.exit(1)

    # ──      VirtualHome    ────────────────────────────────────────────
    _backend_proc = None
    if not args.no_launch:
        print(f"\n{'─' * 60}")
        print(f"VirtualHome backend: {args.backend_exe}")
        _cfg_tmp = load_config(args.config)
        _host = resolve_backend_host(_cfg_tmp)
        _port = resolve_backend_port(_cfg_tmp)
        try:
            _backend_proc = _launch_backend(args.backend_exe, args.backend_args, _port)
        except FileNotFoundError as _exc:
            print(f"❌ {_exc}")
            print("Check config.yaml env.backend_exe or pass --backend-exe.")
            sys.exit(1)

        print(f"Waiting for backend port: {_host}:{_port} (timeout {args.startup_timeout}s)")
        if not _wait_for_port(_host, _port, args.startup_timeout):
            print(f"❌ Backend port did not open within {args.startup_timeout}s: {_host}:{_port}")
            _terminate_backend(_backend_proc)
            sys.exit(1)
        print(f"✓ Backend port ready: {_host}:{_port}")
        print(f"{'─' * 60}\n")
    else:
        print("ℹ️  --no-launch: using an already running VirtualHome backend")
        print("⚠️  Make sure the backend scene matches this task")

    try:
        _run_evaluate(args)
    finally:
        if _backend_proc is not None and not args.keep_backend:
            print("\nStopping VirtualHome backend...")
            _terminate_backend(_backend_proc)
            print("✓ Backend stopped")
        elif _backend_proc is not None and args.keep_backend:
            print("\nℹ️  --keep-backend: backend left running")


def _run_evaluate(args):
    """    evaluate   （  main      ，    try/finally     ） """
    #     
    config = load_config(args.config)

    #       
    task_config = None
    if args.task:
        #   1：    tasks/      task.json
        task_folder = os.path.join("tasks", args.task)
        task_json_path = os.path.join(task_folder, "task.json")
        if os.path.exists(task_json_path):
            print(f"✓ Loaded task file: {task_json_path}")
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_config = json.load(f)
        else:
            #   2：  config.yaml   task_presets   
            try:
                task_config = config.apply_task_by_name(args.task)
                print(f"✓ Loaded task preset: {args.task}")
            except (ValueError, AttributeError) as e:
                print(f"❌ Failed to load task '{args.task}': {e}")
                print("Available presets: ", end="")
                try:
                    presets = config.get("task_presets", {})
                    print(", ".join(presets.keys()) if presets else "( )")
                except Exception:
                    print("(    )")
                sys.exit(1)

    if task_config is None:
        task_config = {
            "name": "custom_action_sequence",
            "description": "Execute custom action sequence",
            "max_steps": 50,
        }

    #     
    scene = args.scene
    if scene is None:
        scene = task_config.get("scene", 0)
        if isinstance(scene, str):
            try:
                scene = int(scene)
            except ValueError:
                scene = 0

    #       （   ：golden_actions > --actions > --action-file）
    action_sequence = None

    #    1：         golden_actions
    if args.task:
        action_sequence = extract_golden_actions_from_task(task_config)

    #    2：    --actions   
    if args.actions:
        print("✓ Loaded actions from --actions (overrides golden_actions)")
        action_sequence = parse_action_sequence(args.actions)

    #    3：     
    if action_sequence is None and args.action_file:
        print(f"✓ Loading actions from file: {args.action_file}")
        try:
            with open(args.action_file, "r", encoding="utf-8") as f:
                action_str = f.read().strip()
            action_sequence = parse_action_sequence(action_str)
        except Exception as e:
            print(f"❌ Failed to read action file: {e}")
            sys.exit(1)

    if not action_sequence:
        print("❌ No valid actions found")
        print("   Possible fixes:")
        print("   1. Use --task with task.json golden_actions")
        print("   2. Use --actions with a comma-separated sequence")
        print("   3. Use --action-file with a sequence file")
        sys.exit(1)

    #    init.json（    ）
    init_action_strings = []
    init_char_position = None
    init_char_rotation = None
    init_char_yaw_degrees = None
    init_camera_pitch = None
    recorded_user_actions = None
    if args.task:
        task_folder = os.path.join("tasks", args.task)
        if os.path.isdir(task_folder):
            init_file = os.path.join(task_folder, "init.json")
            loaded = load_discrete_init_data(init_file, yaw_step_degrees=30.0)
            if loaded.get("error"):
                print(f"⚠️  Failed to load init.json: {loaded['error']}")
            else:
                init_action_strings = loaded.get("init_action_strings", [])
                init_scene = loaded.get("init_scene")
                init_char_position = loaded.get("init_char_position")
                init_char_rotation = loaded.get("init_char_rotation")
                init_char_yaw_degrees = loaded.get("init_char_yaw_degrees")
                init_camera_pitch = loaded.get("init_camera_pitch")
                recorded_user_actions = loaded.get("recorded_user_actions")

                if init_scene is not None:
                    scene = int(init_scene)
                    print(f"✓ init.json scene: {scene}")

                if init_char_position:
                    print(f"✓ init.json character position: {init_char_position}")
                if init_char_yaw_degrees is not None:
                    print(f"✓ init.json character yaw: {init_char_yaw_degrees}°")

                if init_camera_pitch is not None:
                    print(f"✓ init.json camera pitch: {init_camera_pitch}°")

                if init_action_strings and not init_char_position:
                    print(
                        f"✓ Loaded {len(init_action_strings)} init actions from {init_file}"
                    )

    # NOTE: object ID injection is now handled inside execute_action_sequence
    # via env.execute_golden_action_sequence() -> merge_recorded_user_actions().

    #       
    print(f"\nAction sequence ({len(action_sequence)} actions):")
    for i, action in enumerate(action_sequence, 1):
        action_type = action.get("action_type", "?")
        action_str = format_vh_action_dict(action)
        obj_id = action.get("object_id")
        obj2_id = action.get("object2_id")
        binding_suffix = ""
        if obj_id is not None:
            binding_suffix = f" | object_id={obj_id}"
        if obj2_id is not None:
            binding_suffix += f", object2_id={obj2_id}"
        print(f"  {i}. [{action_type}] {action_str}{binding_suffix}")

    #       
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"vh_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    #   
    print_runtime_diagnostics(config)

    #    virtualhome      
    ok_import, _module_path = _try_import_virtualhome_with_fallback()
    if not ok_import:
        print("\n" + "=" * 70)
        print("❌ virtualhome Python package is not available")
        print("=" * 70)
        print("Install option: pip install virtualhome==2.3.0")
        print("Source option: git clone virtualhome && pip install -e .")
        print("=" * 70 + "\n")
        sys.exit(1)

    #    VirtualHome Unity
    try:
        from envs.virtualhome import VirtualHomeEnvWrapper

        env_config = config.get_all()
        if not isinstance(env_config, dict):
            env_config = {}
        env_cfg = env_config.get("env")
        if not isinstance(env_cfg, dict):
            env_cfg = {}
            env_config["env"] = env_cfg
        #           /        scene，    config.yaml    scene    
        env_cfg["scene"] = int(scene)

        env = VirtualHomeEnvWrapper(
            scene=scene,
            port=str(config.get("env.port", "8080")),
            url=config.get("env.url", "127.0.0.1"),
            width=config.get("env.width", 640),
            height=config.get("env.height", 480),
            output_dir=output_dir,
            config=env_config,
        )
        env.require_visible_for_interaction = True
        env.require_close_for_interaction = True
        print(
            "ℹ️  Strict interaction checks enabled "
            "(require_visible_for_interaction=True, require_close_for_interaction=True)"
        )
    except ImportError as e:
        print(f"❌ Failed to import VirtualHome package: {e}")
        print("💡 Hint: pip install virtualhome==2.3.0")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to connect to VirtualHome Unity backend: {e}")
        print("💡 Hint: make sure the VirtualHome Unity backend is running")
        sys.exit(1)

    #   task_config     （success_conditions target_object_types）
    #    wrapper   reset()   step()            
    _configure_env_from_task(env, task_config)

    try:
        result = execute_action_sequence(
            env,
            action_sequence,
            task_config,
            output_dir,
            scene=scene,
            init_action_strings=(
                init_action_strings if not init_char_position else None
            ),
            step_by_step=args.step_by_step,
            char_position=init_char_position,
            char_rotation=init_char_rotation,
            char_yaw_degrees=init_char_yaw_degrees,
            camera_pitch=init_camera_pitch,
            recorded_user_actions=recorded_user_actions,
            backend_mode=(
                "external_no_launch"
                if args.no_launch
                else "isolated_launched_by_evaluate"
            ),
        )
    finally:
        env.close()

    #       
    print(f"\n{'=' * 60}")
    print("Final Summary")
    print(f"{'=' * 60}")
    print(f"Task name: {result['task_name']}")
    print(f"Steps executed: {result['step_count']}")
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Evaluation score: {result['evaluation_score']:.2f}")

    if result["error_messages"]:
        print(f"Errors ({len(result['error_messages'])}):")
        for msg in result["error_messages"]:
            print(f"  - {msg}")

    print(f"{'=' * 60}")

    #    TSV   
    if args.task and action_sequence:
        print(f"\n{'=' * 80}")
        print("TSV row (paste into Excel/Sheet)")
        print(f"{'=' * 80}")

        try:
            task_id = args.task
            csv_row = generate_csv_row(task_config, action_sequence, task_id)

            field_order = [
                "Task ID",
                "Task Name",
                "Instruction",
                "Golden Action",
                "Step Number",
                "Evaluation",
                "Category",
                "Level",
                "Evaluation Type",
                "Plan",
                "Env",
                "Comment",
                "Annotation",
                "Anotator",
                "Check",
                "Checker",
            ]

            tsv_values = [str(csv_row.get(f, "")) for f in field_order]
            tsv_line = "\t".join(tsv_values)

            print(f"\n")
            print(f'"{tsv_line}"')
            print(f"\n")

            copy_to_clipboard(tsv_line)
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌    TSV     : {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
