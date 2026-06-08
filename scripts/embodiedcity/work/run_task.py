"""
VLN Agent - Vision-Language Navigation Agent for EmbodiedCity Simulator

Uses config file, JSON data, API utils, text-based agent tool prompts.
Actions match workspace/human_check_v2.py: Forward, Backward, Left, Right,
Up, Down, Yaw Left, Yaw Right, plus agent terminal actions Done and Fail.
"""

import os
import sys
import json
import math
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import compute_max_steps_from_n
import glob
import time
import logging
import numpy as np
import cv2
import airsim
from PIL import Image
from datetime import datetime
from openai import OpenAI

import yaml
import importlib.util

# ---------------------------------------------------------------------------
# Module-level logger  (console handler only at startup; per-task file handler
# is added/removed dynamically in run_all so every task gets its own agent.log)
# ---------------------------------------------------------------------------
_log = logging.getLogger("vln_agent")
_log.setLevel(logging.DEBUG)
_log.propagate = False

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(message)s"))
_log.addHandler(_console_handler)


def _add_file_log_handler(log_dir: str) -> logging.FileHandler:
    """Create <log_dir>/agent.log and attach a FileHandler to _log."""
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, "agent.log"), mode="a", encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    _log.addHandler(fh)
    return fh


def _remove_file_log_handler(fh: logging.FileHandler) -> None:
    """Flush, close, and detach a FileHandler from _log."""
    fh.flush()
    fh.close()
    _log.removeHandler(fh)

# Load utils/api_utils.py directly (the directory has no __init__.py)
_api_utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "api_utils.py")
_spec = importlib.util.spec_from_file_location("api_utils", _api_utils_path)
_api_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api_utils)

encode_image               = _api_utils.encode_image
create_openai_response     = _api_utils.create_openai_response
build_conversation_history = _api_utils.build_conversation_history
add_response_to_history    = _api_utils.add_response_to_history
get_response_text          = _api_utils.get_response_text
parse_json_from_text       = _api_utils.parse_json_from_text
apply_history_window       = _api_utils.apply_history_window


def apply_history_window(history: list, window):
    """
    Keep the first system message, plus enough recent user/assistant turns and
    the current user message so the total number of user-image messages sent to
    the model is at most N, including the current step.
    """
    if window is None or len(history) <= 1:
        return history

    window = int(window)
    first_message = history[:1]
    remaining = history[1:]

    current_user = []
    completed_turns = remaining
    if remaining and remaining[-1].get("role") == "user":
        current_user = [remaining[-1]]
        completed_turns = remaining[:-1]

    if window <= 0:
        return first_message + current_user

    history_turns_to_keep = max(window - len(current_user), 0)
    return first_message + completed_turns[-2 * history_turns_to_keep:] + current_user

# ---------------------------------------------------------------------------
# Action definitions  (must match workspace/human_check_v2.py key bindings)
# ---------------------------------------------------------------------------
ACTION_FORWARD   = "forward"
ACTION_BACKWARD  = "backward"
ACTION_LEFT      = "left"
ACTION_RIGHT     = "right"
ACTION_UP        = "up"
ACTION_DOWN      = "down"
ACTION_YAW_LEFT  = "yaw_left"
ACTION_YAW_RIGHT = "yaw_right"
ACTION_DONE      = "done"
ACTION_FAIL      = "fail"

NAVIGATION_ACTIONS = {
    ACTION_FORWARD,
    ACTION_BACKWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_YAW_LEFT,
    ACTION_YAW_RIGHT,
}
TERMINAL_ACTIONS = {ACTION_DONE, ACTION_FAIL}
VALID_ACTIONS = NAVIGATION_ACTIONS | TERMINAL_ACTIONS
ACTION_MATCH_ORDER = sorted(VALID_ACTIONS, key=len, reverse=True)

# Actions that accept granularity (movement/rotation). Terminal actions have none.
MOVEMENT_ACTIONS = {
    ACTION_FORWARD,
    ACTION_BACKWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN,
}
ROTATION_ACTIONS = {ACTION_YAW_LEFT, ACTION_YAW_RIGHT}
GRANULARITY_ACTIONS = NAVIGATION_ACTIONS
GRANULARITY_LEVELS  = {"small", "medium", "large"}
ROTATION_GRANULARITY_LEVELS = {"medium", "large"}
DEFAULT_GRANULARITY = "medium"
TERMINAL_GRANULARITY = "none"

ACTIONS_DESCRIPTION = """\
Movement actions (support granularity: small | medium | large):
- forward:   Move forward along the current heading direction.
- backward:  Move backward, opposite to the current heading direction.
- left:      Strafe sideways to the left (no heading change).
- right:     Strafe sideways to the right (no heading change).
- up:        Increase altitude (move upward).
- down:      Decrease altitude (move downward).

Rotation actions (support granularity: medium | large only):
- yaw_left:  Rotate counterclockwise (turn left in place).
- yaw_right: Rotate clockwise (turn right in place).

Terminal actions (do not move and do not use granularity):
- DONE:      Indicates that you believe the task has been successfully completed. Use only after you have verified that all task goals are satisfied.
- FAIL:      Indicates that you believe the task cannot be completed or you refuse to continue. Use only when the task is impossible to complete or an unrecoverable situation is encountered.

"""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tasks(cfg: dict) -> list:
    """
    Scan cfg['data']['root'] for JSON files and return a list of task dicts.
    Each task dict preserves all original fields from the JSON.
    """
    root = cfg["data"]["root"]
    pattern = cfg["data"].get("pattern", "**/*.json")
    max_tasks = cfg["data"].get("max_tasks", None)

    file_paths = glob.glob(os.path.join(root, pattern), recursive=True)
    file_paths = sorted(file_paths)

    tasks = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file_path"] = fp
            tasks.append(data)
        except Exception as e:
            _log.warning(f"[WARN] Failed to load {fp}: {e}")

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def _load_task_ids_from_file(path: str) -> list[str]:
    """Load task ids from JSON ([..] or {\"task_ids\": [...]}) or one id per line."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "task_ids" in data:
            return [str(x).strip() for x in data["task_ids"] if str(x).strip()]
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# AirSim movement helpers  (teleport-style, matching human_check_v2.py)
# ---------------------------------------------------------------------------

def _get_yaw_from_quat(q) -> float:
    """Return yaw in degrees from an AirSim quaternion."""
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _get_state(client, vehicle_name: str):
    """Return (position_Vector3r, orientation_Quaternionr) of the drone."""
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    return pos, ori


def _teleport(client, vehicle_name: str, target_pos, target_quat):
    """Teleport the drone to the target pose instantly (no physics delay)."""
    client.simSetVehiclePose(
        airsim.Pose(target_pos, target_quat),
        True,
        vehicle_name=vehicle_name,
    )
    client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=vehicle_name)


def _resolve_granularity(action: str, granularity: str, gran_cfg: dict) -> tuple:
    """
    Return (move_dist, rotate_angle) for the action and granularity level.
    gran_cfg is the navigation.granularity block from config.
    """
    if action in ROTATION_ACTIONS:
        rotate_key = "angle_large" if granularity == "large" else "angle_medium"
        return gran_cfg["dist_medium"], gran_cfg[rotate_key]

    if granularity == "small":
        move_key = "dist_small"
    elif granularity == "large":
        move_key = "dist_large"
    else:
        move_key = "dist_medium"
    return gran_cfg[move_key], gran_cfg["angle_medium"]


def execute_action(
    client,
    vehicle_name: str,
    action: str,
    granularity: str,
    gran_cfg: dict,
) -> None:
    """
    Execute one navigation action via teleport-style pose update.
    Movement actions support small/medium/large distances. Rotation actions
    support only medium/large angles.
    """
    move_dist, rotate_angle = _resolve_granularity(action, granularity, gran_cfg)

    pos, ori = _get_state(client, vehicle_name)
    yaw_deg = _get_yaw_from_quat(ori)
    yaw_rad = math.radians(yaw_deg)

    new_x = pos.x_val
    new_y = pos.y_val
    new_z = pos.z_val
    new_yaw_deg = yaw_deg

    if action == ACTION_FORWARD:
        new_x += move_dist * math.cos(yaw_rad)
        new_y += move_dist * math.sin(yaw_rad)
    elif action == ACTION_BACKWARD:
        new_x -= move_dist * math.cos(yaw_rad)
        new_y -= move_dist * math.sin(yaw_rad)
    elif action == ACTION_LEFT:
        new_x += move_dist * math.sin(yaw_rad)
        new_y -= move_dist * math.cos(yaw_rad)
    elif action == ACTION_RIGHT:
        new_x -= move_dist * math.sin(yaw_rad)
        new_y += move_dist * math.cos(yaw_rad)
    elif action == ACTION_UP:
        new_z -= move_dist          # AirSim: negative Z = up
    elif action == ACTION_DOWN:
        new_z += move_dist
    elif action == ACTION_YAW_LEFT:
        new_yaw_deg -= rotate_angle
    elif action == ACTION_YAW_RIGHT:
        new_yaw_deg += rotate_angle
    else:
        return                      # unknown: do nothing

    target_pos = airsim.Vector3r(new_x, new_y, new_z)
    target_quat = airsim.to_quaternion(0, 0, math.radians(new_yaw_deg))
    _teleport(client, vehicle_name, target_pos, target_quat)


# ---------------------------------------------------------------------------
# Image capture
# ---------------------------------------------------------------------------

def capture_image(client, vehicle_name: str) -> Image.Image:
    """Capture front camera RGB image and return as PIL Image."""
    responses = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
        vehicle_name=vehicle_name,
    )
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = img1d.reshape(response.height, response.width, 3)
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    return Image.fromarray(img_rgb)


# ---------------------------------------------------------------------------
# Prompt builder  (text-based agent-tool format)
# ---------------------------------------------------------------------------

def build_system_prompt(instruction: str, gran_cfg: dict) -> str:
    d_s = gran_cfg["dist_small"]
    d_m = gran_cfg["dist_medium"]
    d_l = gran_cfg["dist_large"]
    a_m = gran_cfg["angle_medium"]
    a_l = gran_cfg["angle_large"]
    return f"""\
You are an autonomous navigation agent controlling a drone in a 3D urban simulation.
Your goal is to navigate to the destination described in the task instruction.

TASK INSTRUCTION:
{instruction}

SUCCESS CRITERION:
The task is successful only if you explicitly output the terminal action "done" while the drone is within 5 meters of the destination described above.
If you output "done" before reaching the destination, the task immediately fails and this trajectory ends.
If you output "fail", the task immediately fails and this trajectory ends.
If the maximum number of steps is exceeded before a successful "done", the task fails.

AVAILABLE ACTIONS:
{ACTIONS_DESCRIPTION}

GRANULARITY:
- Movement actions (forward/backward/left/right/up/down):
  - small:  move {d_s} m
  - medium: move {d_m} m
  - large:  move {d_l} m
- Rotation actions (yaw_left/yaw_right):
  - medium: rotate {a_m} deg
  - large:  rotate {a_l} deg
  Do not use small for rotation actions.

At every step you will receive the current camera view image.
Based on the visual observations, choose the most appropriate next action and granularity so that you approach the destination and end up within 5 meters of it.
When you believe you have reached the destination, output "done". If you determine the destination cannot be reached, output "fail".

First, reason about what you observe and what action to take.
Then output a JSON object with exactly this structure:
{{
  "action": "<one of the action names listed above>",
  "granularity": "<small | medium | large for movement; medium | large for rotation; none for done/fail>"
}}"""


def build_step_prompt(step: int) -> str:
    return (
        f"Current step: {step}.\n"
        "Current view image is attached. Choose your next action.\n"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> tuple:
    """
    Parse the model response and return (action_name, granularity).
    Terminal actions return TERMINAL_GRANULARITY.
    Navigation granularity defaults to DEFAULT_GRANULARITY if missing or invalid.
    Falls back to forward/medium if parsing fails entirely.
    """
    json_data = parse_json_from_text(response_text)
    if json_data and "action" in json_data:
        action = str(json_data["action"]).strip().lower()
        raw_gran = str(json_data.get("granularity") or DEFAULT_GRANULARITY).strip().lower()
        granularity = raw_gran if raw_gran in GRANULARITY_LEVELS else DEFAULT_GRANULARITY

        if action in TERMINAL_ACTIONS:
            return action, TERMINAL_GRANULARITY
        if action in VALID_ACTIONS:
            if action in ROTATION_ACTIONS and granularity not in ROTATION_GRANULARITY_LEVELS:
                return ACTION_FORWARD, DEFAULT_GRANULARITY
            return action, granularity
        # Try partial match
        for valid in ACTION_MATCH_ORDER:
            if valid in action:
                if valid in TERMINAL_ACTIONS:
                    return valid, TERMINAL_GRANULARITY
                if valid in ROTATION_ACTIONS and granularity not in ROTATION_GRANULARITY_LEVELS:
                    return ACTION_FORWARD, DEFAULT_GRANULARITY
                return valid, granularity

    # Plain-text fallback: scan for known action keywords.
    text_lower = str(response_text).lower()
    gran_found = DEFAULT_GRANULARITY
    for lvl in GRANULARITY_LEVELS:
        if lvl in text_lower:
            gran_found = lvl
            break
    for act in ACTION_MATCH_ORDER:
        if act in text_lower:
            if act in TERMINAL_ACTIONS:
                return act, TERMINAL_GRANULARITY
            if act in ROTATION_ACTIONS and gran_found not in ROTATION_GRANULARITY_LEVELS:
                return ACTION_FORWARD, DEFAULT_GRANULARITY
            return act, gran_found

    return ACTION_FORWARD, DEFAULT_GRANULARITY



# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

class StepRecord:
    def __init__(self, step: int, action: str, granularity: str,
                 raw_response: str, image: Image.Image, pos_x: float, pos_y: float,
                 pos_z: float, yaw_deg: float):
        self.step = step
        self.action = action
        self.granularity = granularity
        self.raw_response = raw_response
        self.image = image
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.yaw_deg = yaw_deg

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "granularity": self.granularity,
            "raw_response": self.raw_response,
            "position": [self.pos_x, self.pos_y, self.pos_z],
            "yaw_deg": self.yaw_deg,
        }

    def to_history_line(self) -> str:
        gran_tag = f"({self.granularity})"
        return (
            f"  Step {self.step:2d}: {self.action:<10s}{gran_tag:<8s} | "
            f"pos=({self.pos_x:.1f}, {self.pos_y:.1f}, {self.pos_z:.1f}) | "
            f"yaw={self.yaw_deg:.0f}deg"
        )


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _write_video(frames: list, video_path: str, fps: int) -> None:
    """Write a list of PIL Images as an MP4 video using cv2."""
    if not frames:
        return
    # Determine frame size from the first valid frame
    w, h = frames[0].size          # PIL size is (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for img in frames:
        # PIL RGB -> numpy BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


def save_task_result(
    task: dict,
    records: list,
    final_dist: float,
    success: bool,
    save_dir: str,
    save_images: bool,
    save_video: bool,
    video_fps: int,
) -> None:
    task_id = task.get("id", "unknown")
    task_save_dir = os.path.join(save_dir, task_id)
    os.makedirs(task_save_dir, exist_ok=True)

    valid_records = [r for r in records if r.image is not None]

    # Save per-step PNG images
    if save_images:
        for rec in valid_records:
            img_path = os.path.join(task_save_dir, f"step_{rec.step:03d}_{rec.action}.png")
            rec.image.save(img_path)

    # Save first-person view video
    if save_video and valid_records:
        video_path = os.path.join(task_save_dir, "fpv.mp4")
        _write_video([r.image for r in valid_records], video_path, video_fps)
        _log.info(f"  [video] {video_path}")

    # Save JSON history
    result = {
        "id": task_id,
        "instruction": task.get("instruction", ""),
        "success": success,
        "final_dist": round(final_dist, 3),
        "total_steps": len(records),
        "trajectory": [r.to_dict() for r in records],
    }
    result_path = os.path.join(task_save_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _log.info(f"  [saved] {task_save_dir}")


# ---------------------------------------------------------------------------
# VLN Agent  (main class)
# ---------------------------------------------------------------------------

class VLNAgent:
    def __init__(self, config_path: str, model: str = None, api_key: str = None, api_base: str = None,
                 output_dir: str = None, task_ids: list = None,
                 temperature: float = None, top_p: float = None,
                 history_window: int = None):
        self.cfg = load_config(config_path)
        self.output_dir = output_dir  # CLI override; None means use config
        self.task_ids = task_ids  # None = run all loaded tasks; else filter by task["id"]
        self.history_window = history_window
        self._init_client(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
        )
        self._init_airsim()

    def _init_client(
        self,
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        temperature: float = None,
        top_p: float = None,
    ):
        model_cfg = self.cfg["model"]
        self.model_name = model or model_cfg["name"]
        self.temperature = float(
            temperature if temperature is not None else model_cfg.get("temperature", 1.0)
        )
        self.top_p = float(top_p if top_p is not None else model_cfg.get("top_p", 0.9))
        self.extra_body = model_cfg.get("extra_body")
        self.llm_client = OpenAI(
            api_key=api_key or model_cfg["api_key"],
            base_url=api_base or model_cfg.get("api_base", None),
        )

    def _init_airsim(self):
        self.vehicle_name = "Drone1"
        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.airsim_client.armDisarm(True, vehicle_name=self.vehicle_name)
        try:
            self.airsim_client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            self.airsim_client.moveToZAsync(-2, 1, vehicle_name=self.vehicle_name).join()
        except Exception as e:
            _log.warning(f"[WARN] Takeoff failed (may already be airborne): {e}")

    def _teleport_to_start(self, task: dict) -> None:
        start = task["start"]
        start_yaw = math.radians(task.get("start_yaw", 0.0))
        pos = airsim.Vector3r(start[0], start[1], start[2])
        quat = airsim.to_quaternion(0, 0, start_yaw)
        self.airsim_client.simSetVehiclePose(
            airsim.Pose(pos, quat), True, vehicle_name=self.vehicle_name
        )
        self.airsim_client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=self.vehicle_name)
        time.sleep(3)  

    def _dist_to_end(self, task: dict) -> float:
        end = task["end"]
        pos, _ = _get_state(self.airsim_client, self.vehicle_name)
        return math.sqrt(
            (pos.x_val - end[0]) ** 2
            + (pos.y_val - end[1]) ** 2
            + (pos.z_val - end[2]) ** 2
        )

    def _query_model(self, conversation_history: list) -> str:
        max_retries = self.cfg.get("api", {}).get("max_retries", 3)
        for attempt in range(1, max_retries + 1):
            try:
                response = create_openai_response(
                    self.llm_client,
                    self.model_name,
                    conversation_history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body=self.extra_body,
                )
                return get_response_text(response)
            except Exception as e:
                _log.warning(f"  [WARN] API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)

    def run_task(self, task: dict) -> dict:
        nav_cfg = self.cfg["navigation"]
        _max_steps_cfg = nav_cfg.get("max_steps", 30)
        if str(_max_steps_cfg).lower() == "dynamic":
            n = len(task.get("trajectory", []))
            max_steps = compute_max_steps_from_n(n)
            _log.info(f"  [dynamic] trajectory len={n}, max_steps={max_steps} (10+2n)")
        else:
            max_steps = int(_max_steps_cfg)
        stop_threshold  = nav_cfg.get("stop_threshold", 5.0)
        history_window  = (
            self.history_window if self.history_window is not None
            else nav_cfg.get("history_window", None)
        )
        gran_cfg        = nav_cfg.get("granularity", {
            "dist_small": 2.5,  "dist_medium": 5.0,  "dist_large": 10.0,
            "angle_medium": 30, "angle_large": 90,
        })

        task_id     = task.get("id", "unknown")
        instruction = task.get("instruction", "")

        _log.info(f"\n{'='*60}")
        _log.info(f"Task: {task_id}")
        _log.info(f"Instruction: {instruction}")
        _log.info(f"{'='*60}")

        # Teleport to start position
        self._teleport_to_start(task)
        time.sleep(0.3)

        system_prompt = build_system_prompt(instruction, gran_cfg)
        conversation_history = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        ]
        records: list[StepRecord] = []

        success = False
        api_error = False
        for step in range(1, max_steps + 1):
            pos, ori = _get_state(self.airsim_client, self.vehicle_name)
            yaw_deg = _get_yaw_from_quat(ori)

            # Capture image
            try:
                image = capture_image(self.airsim_client, self.vehicle_name)
            except Exception as e:
                _log.warning(f"  [WARN] Image capture failed at step {step}: {e}")
                image = None

            # Build prompt for this step
            step_prompt = build_step_prompt(step)

            # Encode image to base64
            img_b64 = encode_image(image) if image is not None else ""

            # Add user message with the per-step prompt and current image
            user_content = [{"type": "input_text", "text": step_prompt}]
            if img_b64:
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img_b64}",
                })
            conversation_history.append({"role": "user", "content": user_content})

            # Query model
            try:
                response_text = self._query_model(apply_history_window(conversation_history, history_window))
            except Exception as e:
                _log.error(f"  [ERROR] API retries exhausted at step {step}: {e}. Skipping task.", exc_info=True)
                api_error = True
                break
            _log.info(f"  Step {step} | Model respons.")

            # Add assistant response to history
            conversation_history = add_response_to_history(conversation_history, response_text)

            # Parse action + granularity
            action, granularity = parse_action(response_text)
            _log.info(f"  Step {step} | Action: {action} ({granularity})")

            # Record step (before executing so image matches pre-action view)
            rec = StepRecord(
                step=step,
                action=action,
                granularity=granularity,
                raw_response=response_text,
                image=image,
                pos_x=pos.x_val,
                pos_y=pos.y_val,
                pos_z=pos.z_val,
                yaw_deg=yaw_deg,
            )
            records.append(rec)

            if action == ACTION_DONE:
                dist = self._dist_to_end(task)
                _log.info(f"  Step {step} | Done requested | Dist to end: {dist:.2f}")
                if dist < stop_threshold:
                    success = True
                    _log.info(f"  Done accepted. Reached destination: dist={dist:.2f}")
                else:
                    success = False
                    _log.info(f"  Done rejected. Too far from destination: dist={dist:.2f}")
                break

            if action == ACTION_FAIL:
                success = False
                dist = self._dist_to_end(task)
                _log.info(f"  Step {step} | Fail requested | Dist to end: {dist:.2f}")
                break

            # Execute navigation action with granularity.
            execute_action(
                self.airsim_client,
                self.vehicle_name,
                action,
                granularity,
                gran_cfg,
            )
            time.sleep(0.1)

            dist = self._dist_to_end(task)
            _log.info(f"  Step {step} | Dist to end: {dist:.2f}")

        final_dist = self._dist_to_end(task)
        if (
            not success
            and not api_error
            and len(records) >= max_steps
            and (not records or records[-1].action not in TERMINAL_ACTIONS)
        ):
            _log.info(f"  Max steps reached without successful Done. Task failed.")

        _log.info(f"  Task done | success={success} | final_dist={final_dist:.2f} | steps={len(records)}")
        return {
            "task_id": task_id,
            "success": success,
            "final_dist": final_dist,
            "steps": len(records),
            "records": records,
            "api_error": api_error,
        }

    def _has_result(self, task: dict, save_dir: str) -> bool:
        """Return True if a result.json already exists for this task."""
        task_id = task.get("id", "unknown")
        result_path = os.path.join(save_dir, task_id, "result.json")
        return os.path.exists(result_path)

    def run_all(self) -> None:
        out_cfg = self.cfg["output"]
        # If output_dir is not set and config has no save_dir, derive it from the model name.
        if self.output_dir:
            save_dir = self.output_dir
        elif "save_dir" in out_cfg and out_cfg["save_dir"]:
            save_dir = out_cfg["save_dir"]
        else:
            # Derive output directory from the model name.
            model_name = self.model_name.replace("/", "_").replace("\\", "_")
            save_dir = f"output/{model_name}"
        save_images = out_cfg.get("save_images", True)
        save_video  = out_cfg.get("save_video", False)
        video_fps   = out_cfg.get("video_fps", 2)
        resume      = self.cfg.get("resume", False)

        tasks = load_tasks(self.cfg)
        if self.task_ids:
            allowed = {x.strip() for x in self.task_ids if x and str(x).strip()}
            before = len(tasks)
            tasks = [t for t in tasks if t.get("id") in allowed]
            found_ids = {t.get("id") for t in tasks}
            missing = allowed - found_ids
            if missing:
                _log.warning(f"[WARN] No matching task JSON for id(s): {sorted(missing)}")
            _log.info(f"Loaded {before} tasks from disk; --tasks filter -> running {len(tasks)} task(s).")
        else:
            _log.info(f"Loaded {len(tasks)} tasks.")

        os.makedirs(save_dir, exist_ok=True)

        def _run_batch(task_list: list) -> None:
            for task in task_list:
                task_id = task.get("id", "unknown")
                task_log_dir = os.path.join(save_dir, task_id)
                fh = _add_file_log_handler(task_log_dir)
                try:
                    result = self.run_task(task)
                    if result["api_error"]:
                        _log.info(f"  [skip save] Task {result['task_id']} aborted due to API error; result not saved for resume.")
                    else:
                        save_task_result(
                            task,
                            result["records"],
                            result["final_dist"],
                            result["success"],
                            save_dir,
                            save_images,
                            save_video,
                            video_fps,
                        )
                except Exception as e:
                    _log.error(f"  [ERROR] Unexpected error in task {task_id}: {e}", exc_info=True)
                finally:
                    _remove_file_log_handler(fh)

        if resume:
            pass_num = 0
            while True:
                pending = [t for t in tasks if not self._has_result(t, save_dir)]
                if not pending:
                    _log.info(f"[resume] All {len(tasks)} tasks have results. Done.")
                    break
                pass_num += 1
                _log.info(f"[resume] Pass {pass_num}: {len(pending)}/{len(tasks)} tasks pending.")
                _run_batch(pending)
        else:
            _run_batch(tasks)

        # Build summary: only tasks in current task list that have result.json
        task_ids = {t.get("id") for t in tasks}
        pending_final = [t for t in tasks if not self._has_result(t, save_dir)]

        summary = []
        for task_id in sorted(task_ids):
            result_path = os.path.join(save_dir, task_id, "result.json")
            if os.path.isfile(result_path):
                try:
                    with open(result_path, encoding="utf-8") as f:
                        r = json.load(f)
                    summary.append({
                        "task_id": r.get("id", task_id),
                        "success": r.get("success", False),
                        "final_dist": r.get("final_dist", 0.0),
                        "steps": r.get("total_steps", 0),
                    })
                except Exception as e:
                    _log.warning(f"[WARN] Failed to read {result_path}: {e}")

        sr = sum(1 for s in summary if s["success"]) / max(len(summary), 1)
        avg_dist = sum(s["final_dist"] for s in summary) / max(len(summary), 1)
        _log.info(f"\n{'='*60}")
        _log.info(f"Results: SR={sr:.3f}  AvgDist={avg_dist:.2f}  Tasks={len(summary)}/{len(tasks)}")

        if pending_final:
            _log.info(f"[skip summary] {len(pending_final)} task(s) still pending; summary not saved.")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(save_dir, f"summary_{ts}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            _log.info(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLN Agent for EmbodiedCity")
    parser.add_argument("--config", type=str, default="experiments/configs/embodiedcity/vln-agent-config-gpt54.yaml",
                        help="Path to the YAML config file")
    parser.add_argument("--model",    type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--api-key",  type=str, default=None, help="API key (overrides config)")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL (overrides config)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides config)")
    parser.add_argument("--top-p", type=float, default=None, dest="top_p", help="Sampling top-p (overrides config)")
    parser.add_argument("--history-window", type=int, default=None, help="Conversation history window (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help="Only run these task id(s), e.g. embodiedcity_5002 embodiedcity_5003 embodiedcity_5008",
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="JSON file with task ids: [..] or {\"task_ids\": [..]}; or one id per line. Overrides --tasks if set.",
    )
    args = parser.parse_args()

    task_ids = args.tasks
    if args.tasks_file:
        task_ids = _load_task_ids_from_file(args.tasks_file)

    agent = VLNAgent(
        args.config,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        output_dir=args.output_dir,
        task_ids=task_ids,
        temperature=args.temperature,
        top_p=args.top_p,
        history_window=args.history_window,
    )
    agent.run_all()
