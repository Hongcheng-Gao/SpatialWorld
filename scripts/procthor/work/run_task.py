"""
ProcTHOR    ：  spatial-planning ProcTHOR runner   
-    config（YAML + tasks/<task_id>/task.json，10+2n）
- get_vlm()       ai2thor   
- Agent   ：API                 ai2thor think_node   ；  log.json + episode_*.json
"""

import os
import sys
import json
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import resolve_max_steps_from_task

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from core.llm.provider import get_vlm
from mllm_base_agent.prompts.procthor import get_procthor_prompt
from mllm_base_agent.prompts.procthor_continuous import get_procthor_continuous_prompt
from core.response_parser import parse_vlm_response
from envs.procthor_wrapper import ProcTHOREnvWrapper
from evaluation.procthor.base import create_evaluator_from_config
from scripts.evaluate_actions_procthor import load_init_actions_for_task

#   spatial-planning core/agent/graph.py LOCAL_RETRY_CONFIG   
LOCAL_RETRY_CONFIG = {
    "max_retries": 3,
    "api_max_retries": 5,
    "retry_delay": 2,
    "api_retry_delay": 5,
}
MODEL_HISTORY_TURNS = 29


class APIRetryError(Exception):
    """API     ，failure_type=api_error"""
    pass


class ParseRetryError(Exception):
    """    （  spatial-planning   ）"""
    pass


def load_init_actions_from_task_folder(task_folder_path: str):
    """        init     （  ），  env.reset    """
    task_file = os.path.join(task_folder_path, "task.json")
    if not os.path.isfile(task_file):
        return None
    return load_init_actions_for_task(task_file)


def perform_final_evaluation(env, task_config: dict, observation) -> tuple:
    """DONE        ，  spatial-planning evaluate_node    """
    if not task_config or not observation or not getattr(observation, "metadata", None):
        return False, 0.0
    try:
        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return (score >= 1.0, score)
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False, 0.0


def save_conversation_log(state: dict, output_dir: str):
    """   log.json，    spatial-planning main.save_conversation_log   （  reward error_message） """
    log_file = os.path.join(output_dir, "log.json")
    conversation_json = {
        "metadata": {
            "task_description": state.get("task_prompt", ""),
            "task_result": "success" if state.get("success") else "failure",
            "fail_reason": state.get("fail_reason"),
            "failure_type": state.get("failure_type"),
            "total_steps": state.get("step_count", 0),
            "max_steps": state.get("max_steps", 0),
        },
        "messages": [],
        "images": [],
    }
    for i, entry in enumerate(state.get("structured_trajectory", [])):
        step_id = entry.get("step", i + 1)
        image_path = entry.get("image_path", "")
        raw_response = entry.get("raw_response", "")
        action_string = entry.get("action_string", "")
        reward = entry.get("reward", 0)
        error_message = entry.get("error_message")
        conversation_json["messages"].append({
            "role": "user",
            "content": f"Step {step_id}/{state.get('max_steps', 30)}" + ("\n<image>" if image_path else ""),
            "step": step_id,
            "image_path": image_path,
        })
        conversation_json["messages"].append({
            "role": "assistant",
            "content": raw_response,
            "step": step_id,
            "action_executed": action_string,
            "reward": reward,
            "error_message": error_message,
        })
        if image_path:
            conversation_json["images"].append(image_path)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(conversation_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Conversation log saved: {log_file}")


def save_episode_log(state: dict, output_dir: str, env) -> None:
    """   episode_*.json，  spatial-planning final_node      """
    scene_name = "ProcTHOR"
    if hasattr(env, "scene") and env.scene is not None:
        scene_name = str(env.scene) if not isinstance(env.scene, dict) else env.scene.get("sceneName", "ProcTHOR")
    scene_short = scene_name.replace(" ", "_").replace("/", "_")[:50]
    task_name = (state.get("config") or {}).get("task", {}).get("name", "task") or "task"
    task_short = task_name.replace(" ", "_").replace("/", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{scene_short}_{task_short}_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)
    simplified_trajectory = []
    for entry in state.get("structured_trajectory", []):
        simplified_trajectory.append({
            "step": entry.get("step"),
            "thinking": entry.get("thinking"),
            "action_string": entry.get("action_string"),
            "reward": entry.get("reward"),
            "error_message": entry.get("error_message"),
        })
    action_sequence = env.get_action_sequence() if hasattr(env, "get_action_sequence") else "(no action records)"
    episode_log = {
        "task": state.get("task_prompt", ""),
        "scene": scene_name,
        "success": state.get("success", False),
        "fail_reason": state.get("fail_reason"),
        "failure_type": state.get("failure_type"),
        "step_count": state.get("step_count", 0),
        "max_steps": state.get("max_steps", 0),
        "action_sequence": action_sequence,
        "trajectory": simplified_trajectory,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "total_reward": sum(
                step.get("reward", 0) or 0
                for step in state.get("structured_trajectory", [])
            ),
            "parse_errors_count": sum(
                1 for step in state.get("structured_trajectory", [])
                if step.get("parse_error") is not None
            ),
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(episode_log, f, ensure_ascii=False, indent=2)
    print(f"✓ Episode log saved: {filepath}")


def run_agent_loop(env, vlm, task_config: dict, task_output_dir: str, config: dict, max_steps_override: int | None = None):
    """    agent   ：Think → Act → Evaluate，  ai2thor   （  API/  /    ） """
    task_prompt = task_config.get("instruction") or task_config.get("description") or "Complete the task."
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else resolve_max_steps_from_task(task_config, int(task_config.get("max_steps", 30)))
    )
    enable_summary = config.get("context_management", {}).get("enable_long_term_summary", False)
    navigation_mode = config.get("actions", {}).get("navigation_mode", "discrete")
    if navigation_mode == "continuous":
        system_prompt = get_procthor_continuous_prompt().format(task_prompt=task_prompt)
    else:
        system_prompt = get_procthor_prompt(enable_summary=enable_summary).format(task_prompt=task_prompt)
    max_retries = LOCAL_RETRY_CONFIG.get("max_retries", 3)
    api_max_retries = LOCAL_RETRY_CONFIG.get("api_max_retries", 5)
    retry_delay = LOCAL_RETRY_CONFIG.get("retry_delay", 2)
    api_retry_delay = LOCAL_RETRY_CONFIG.get("api_retry_delay", 5)

    observation = env.reset(task_prompt)
    step_count = 0
    structured_trajectory = []
    short_term_history = []
    ctx_mgmt = config.get("context_management") or {}
    configured_history = int(ctx_mgmt.get("short_term_history_window_size", MODEL_HISTORY_TURNS))
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
    #     N        ；       +assistant   ，           PNG base64            base64 decode fail
    _img_turns = ctx_mgmt.get("multimodal_history_image_turns")
    multimodal_image_turns = max_history if _img_turns is None else min(max_history, max(0, int(_img_turns)))
    consecutive_failures = 0
    success = False
    fail_reason = None
    failure_type = None

    while step_count < max_steps:
        step_count += 1
        print(f"\n{'=' * 60}\n🧠 Step {step_count}/{max_steps}\n{'=' * 60}")

        from mllm_base_agent.llm.messages import SystemMessage, HumanMessage, AIMessage

        messages = [SystemMessage(content=system_prompt)]
        history_entries = short_term_history[-max_history:]
        n_hist = len(history_entries)
        for idx, entry in enumerate(history_entries):
            step_id = entry.get("step")
            img_path = entry.get("image_path")
            raw = entry.get("raw_response", "")
            include_image = multimodal_image_turns > 0 and idx >= (n_hist - multimodal_image_turns)
            content = []
            if include_image and img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            step_text = f"Step {step_id}"
            if not include_image:
                step_text += (
                    " (image omitted in API context to limit payload; the assistant message below refers to that step)"
                )
            content.append({"type": "text", "text": step_text})
            messages.append(HumanMessage(content=content))
            messages.append(AIMessage(content=raw))

        image_path = observation.image_path
        image_data = None
        for img_attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                break
            except Exception as e:
                print(f"⚠️  Image read error (attempt {img_attempt + 1}/{max_retries}): {e}")
                if img_attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to read image after {max_retries} attempts")
                    fail_reason = f"Image read error: {str(e)}"
                    failure_type = "env_error"
                    structured_trajectory.append({
                        "step": step_count, "thinking": "", "action_string": "", "action": {},
                        "raw_response": "", "parse_error": str(e), "failure_type": failure_type, "image_path": image_path,
                    })
                    break
        if image_data is None:
            break

        current_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            {"type": "text", "text": f"Current step is Step {step_count}. Please output <THINK> and <ACTION> according to the rules above and the current observation."},
        ]
        messages.append(HumanMessage(content=current_content))

        # Stage 1: API retry (  ai2thor think_node   )
        response_text = None
        last_error = None
        for api_attempt in range(api_max_retries):
            try:
                print(f"📡 Calling VLM... (API attempt {api_attempt + 1}/{api_max_retries})")
                response = vlm.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)
                break
            except Exception as api_error:
                err_str = str(api_error)
                is_api = any(k in err_str.lower() for k in ["api", "request failed", "connection", "timeout", "timed out", "http", "429", "500"])
                if "400" in err_str or is_api:
                    print(f"⚠️  API Error (attempt {api_attempt + 1}/{api_max_retries}): {err_str[:200]}")
                    if api_attempt < api_max_retries - 1:
                        delay = api_retry_delay if "400" in err_str else retry_delay
                        print(f"   Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                    last_error = APIRetryError(f"API error after {api_max_retries} attempts: {err_str}")
                    break
                last_error = api_error
                break
        if response_text is None:
            if last_error is None:
                last_error = APIRetryError("Failed to get response from API")
            fail_reason = str(last_error)
            failure_type = "api_error" if isinstance(last_error, APIRetryError) else "env_error"
            structured_trajectory.append({
                "step": step_count, "thinking": "", "action_string": "", "action": {},
                "raw_response": "", "parse_error": str(last_error), "failure_type": failure_type, "image_path": image_path,
            })
            break

        # Stage 2: Parse retry (      VLM      )
        parsed = None
        for parse_attempt in range(max_retries):
            try:
                parsed = parse_vlm_response(
                    response_text,
                    enable_summary=enable_summary,
                    env_type="procthor",
                )
                if parse_attempt > 0:
                    print(f"✓ Success after {parse_attempt + 1} parse attempts")
                break
            except ValueError as e:
                print(f"⚠️  Parse Error (parse attempt {parse_attempt + 1}/{max_retries}): {e}")
                if parse_attempt < max_retries - 1:
                    print(f"   Waiting {retry_delay}s before re-calling VLM...")
                    time.sleep(retry_delay)
                    try:
                        response = vlm.invoke(messages)
                        response_text = response.content if hasattr(response, "content") else str(response)
                    except Exception:
                        pass
                    continue
                fail_reason = f"Parse error: {e}"
                failure_type = "parse_error"
                structured_trajectory.append({
                    "step": step_count, "thinking": "", "action_string": "", "action": {},
                    "raw_response": response_text[:2000], "parse_error": str(e), "failure_type": failure_type, "image_path": image_path,
                })
                break
        if parsed is None:
            break

        action_dict = parsed["parsed_action"]
        action_string = parsed["action_string"]
        thinking_text = parsed["thinking_text"]

        if action_dict.get("action_type") == "task_completion":
            if action_dict.get("action_name") == "DONE":
                success, score = perform_final_evaluation(env, task_config, observation)
                if not success:
                    fail_reason = "Model claimed DONE but success conditions not met"
                print("✅ DONE" if success else "❌ DONE but evaluation failed")
            else:
                success = False
                fail_reason = "Model indicated FAIL"
                print("❌ Model output FAIL")
            structured_trajectory.append({
                "step": step_count, "thinking": thinking_text, "action_string": action_string, "action": action_dict,
                "raw_response": response_text, "image_path": image_path, "reward": 10.0 if success else 0,
            })
            break

        try:
            observation, error_message = env.step_with_action_dict(action_dict)
        except Exception as e:
            error_message = str(e)
            observation = None
        step_reward = 0.0 if error_message else 0.1
        if error_message:
            consecutive_failures += 1
            print(f"  ⚠️  Action failed: {error_message}")
        else:
            consecutive_failures = 0

        structured_trajectory.append({
            "step": step_count, "thinking": thinking_text, "action_string": action_string, "action": action_dict,
            "raw_response": response_text, "error_message": error_message, "image_path": image_path, "reward": step_reward,
        })
        short_term_history.append({"step": step_count, "image_path": image_path, "raw_response": response_text})

        if consecutive_failures >= 4:
            success = False
            fail_reason = f"Consecutive {consecutive_failures} action failures (early stop)"
            failure_type = "action_error"
            print(f"🛑 Early stop: {consecutive_failures} consecutive failures")
            break
        if observation is None:
            fail_reason = "Environment step returned None"
            failure_type = "env_error"
            break

    if step_count >= max_steps and fail_reason is None:
        success = False
        fail_reason = f"Reached maximum step limit ({max_steps} steps)"

    state = {
        "task_prompt": task_prompt,
        "step_count": step_count,
        "max_steps": max_steps,
        "structured_trajectory": structured_trajectory,
        "success": success,
        "fail_reason": fail_reason,
        "failure_type": failure_type,
        "config": config,
    }
    save_conversation_log(state, task_output_dir)
    save_episode_log(state, task_output_dir, env)
    return state


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ProcTHOR embodied agent (VLM loop, aligned with spatial-planning ProcTHOR runner)")
    parser.add_argument("--config", type=str, default="experiments/configs/procthor/config_close_gpt-5.yaml", help="Config file path")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Task ID(s), e.g. procthor000 procthor600")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (per-task dir when running one task)")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--print-config", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    print(f"\n{'=' * 60}\n🔧 Loading Configuration\n{'=' * 60}\nConfig file: {args.config}")
    config = load_config(args.config)
    if args.print_config:
        from config import print_config
        print_config(config)
        return

    task_names = args.tasks or config.get_all_task_names()
    if not task_names:
        print("❌ No tasks specified and no task names from config")
        return

    output_dir = args.output_dir or config.get("experiment.output_dir", "outputs")
    if args.output_dir and len(task_names) == 1:
        run_output_dir = args.output_dir
    else:
        run_output_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    vlm_config = config.get_section("model").get("vlm", {})
    vlm = get_vlm(
        provider=vlm_config.get("provider", "openai"),
        model_name=vlm_config.get("model_name", "gpt-4o"),
        temperature=vlm_config.get("temperature", 0.2),
        top_p=vlm_config.get("top_p"),
        max_tokens=vlm_config.get("max_tokens", 2000),
        base_url=vlm_config.get("base_url"),
        api_key=vlm_config.get("api_key"),
        proxy_url=vlm_config.get("proxy_url"),
    )

    all_results = []
    for task_idx, task_name in enumerate(task_names, 1):
        print(f"\n{'=' * 60}\n📋 Task {task_idx}/{len(task_names)}: {task_name}\n{'=' * 60}")
        task_config = config.apply_task_by_name(task_name)
        if args.max_steps is not None:
            task_config["max_steps"] = args.max_steps
            print(f"✓ max_steps overridden: {args.max_steps}")

        task_output_dir = os.path.join(run_output_dir, task_name) if len(task_names) > 1 else run_output_dir
        os.makedirs(task_output_dir, exist_ok=True)

        task_folder_path = task_config.get("task_folder_path") or os.path.join("tasks", task_name)
        init_actions = load_init_actions_from_task_folder(task_folder_path)

        full_config = config.get_all()
        full_config["task"] = task_config
        full_config["init_actions"] = init_actions or []

        try:
            env = ProcTHOREnvWrapper(
                scene_index=task_config.get("scene_index", 0),
                output_dir=task_output_dir,
                config=full_config,
                headless=args.headless,
            )
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
            continue

        try:
            state = run_agent_loop(
                env,
                vlm,
                task_config,
                task_output_dir,
                full_config,
                max_steps_override=args.max_steps,
            )
            all_results.append({
                "task_name": task_name,
                "success": state.get("success", False),
                "step_count": state.get("step_count", 0),
                "fail_reason": state.get("fail_reason"),
            })
            print(f"\n📊 Result: {'✅ Success' if state['success'] else '❌ Failure'} | Steps: {state['step_count']}/{state['max_steps']}")
        except Exception as e:
            print(f"❌ Task error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
        finally:
            env.close()

    # Summary
    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 80}\n🎉 All Tasks Completed\n{'=' * 80}")
    print(f"Total: {len(all_results)} | Success: {success_count} | Failure: {len(all_results) - success_count}")
    if all_results:
        print(f"Success Rate: {success_count / len(all_results) * 100:.1f}%")
    print(f"Output: {run_output_dir}\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
