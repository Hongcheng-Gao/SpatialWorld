#!/usr/bin/env python3
"""
OpenAI API
MLLM
"""

import pygame
import sys
import os
import json
import base64
import io
import argparse
import asyncio
import logging
import datetime
import time
from typing import Dict, Any, List, Optional
from PIL import Image
from openai import OpenAI

# Add project paths needed when this script is launched from scripts/game.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GAME_ENV_DIR = os.path.join(REPO_ROOT, "mllm_base_agent", "environments", "game")
GAME_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "game")
for path in (REPO_ROOT, GAME_ENV_DIR, GAME_CONFIG_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# MLLM
from games.rubik_cube.rubik_cube_adapter import RubikCubeGame
from input_sources.pygame_input_source import PygameInputSource
from core.evaluation_engine import EvaluationEngine
from core.data_classes import FrameData, Action
from utils.logger import setup_global_logger, get_logger
from utils.openai_utils import encode_image, create_openai_response, build_conversation_history, add_response_to_history
from utils.action_parser import extract_action_info_from_response, parse_action_from_json_rubic

# 
from rubik_config import OPENAI_CONFIG, SYSTEM_PROMPT, OUTPUT_CONFIG, RUBIK_ACTIONS, GAME_CONFIG, EVALUATION_CONFIG


class RubikOpenAIEvaluator:
    """OpenAI - MLLM"""

    def __init__(self, api_base_url: str, api_key: str, model: str = "gpt-5"):
        """
        

        Args:
            api_base_url: OpenAI APIURL
            api_key: API
            model: 
        """
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model = model
        self.logger = None
        self.engine = None
        self.input_source = None
        self.game = None
        self.conversation_history = []  # 
        self.history_window = EVALUATION_CONFIG.get("history_window", None)  # 
        self.retry_times = EVALUATION_CONFIG.get("retry_times", 3)  # API
        self.model_responses = []  # 
        self.game_completed = False  # 
        self.api_fatal_failure = False  # API
        self.log_dir = "logs"  # run_evaluation
        self.config_name = "simple"  # 

        # 
        self.tools = [
            {
                "type": "function",
                "name": "rotate_front_cw",
                "description": "Rotate the F face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_front_acw",
                "description": "Rotate the F face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_back_cw",
                "description": "Rotate the B face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_back_acw",
                "description": "Rotate the B face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_left_cw",
                "description": "Rotate the L face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_left_acw",
                "description": "Rotate the L face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_right_cw",
                "description": "Rotate the R face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_right_acw",
                "description": "Rotate the R face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_up_cw",
                "description": "Rotate the U face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_up_acw",
                "description": "Rotate the U face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_down_cw",
                "description": "Rotate the D face clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_down_acw",
                "description": "Rotate the D face anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_x_cw",
                "description": "Rotate the entire cube around the X-axis (RIGHT face) clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_x_acw",
                "description": "Rotate the entire cube around the X-axis (RIGHT face) anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_y_cw",
                "description": "Rotate the entire cube around the Y-axis (UP face) clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_y_acw",
                "description": "Rotate the entire cube around the Y-axis (UP face) anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_z_cw",
                "description": "Rotate the entire cube around the Z-axis (FRONT face) clockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "rotate_cube_z_acw",
                "description": "Rotate the entire cube around the Z-axis (FRONT face) anticlockwise",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    

    def build_full_prompt(self, is_first_call: bool) -> str:
        """
        

        Args:
            is_first_call: 

        Returns:
            
        """
        if not is_first_call:
            return self.current_prompt

        # 
        tools_description = ""
        for tool in self.tools:
            if tool.get("type") == "function":
                name = tool.get("name", "")
                description = tool.get("description", "")
                tools_description += f"- {name}: {description}\n"

        if tools_description:
            tools_description = "AVAILABLE ACTIONS:\n\n" + tools_description + "\n"

        # 
        response_format = """
RESPONSE FORMAT:
You must respond with a JSON object containing the action you want to take. The JSON should have the following format:
{
  "action": "function_name",
  "face": "face_letter",
  "direction": "CW_or_ACW"
}

Examples:
- To rotate the front face clockwise: {"action": "rotate_front_cw", "face": "F", "direction": "CW"}
- To rotate the left face anticlockwise: {"action": "rotate_left_acw", "face": "L", "direction": "ACW"}
- To rotate the entire cube around X-axis clockwise: {"action": "rotate_cube_x_cw", "face": "X", "direction": "CW"}

IMPORTANT: You must respond with ONLY the JSON object, no additional text or explanations.
"""

        return self.current_prompt + "\n\n" + tools_description + response_format

    def capture_game_screenshot(self) -> str:
        """
        base64

        Returns:
            base64
        """
        if not self.game:
            raise ValueError("Game not initialized")

        # 
        self.game.render()

        # PygamePIL
        pygame_image = pygame.surfarray.array3d(self.game.screen)
        pygame_image = pygame_image.swapaxes(0, 1)
        pil_image = Image.fromarray(pygame_image)

        # base64
        return encode_image(pil_image)

    def record_model_response(self, frame_number: int, response, action_taken: str):
        """
        

        Args:
            frame_number: 
            response: OpenAI API
            action_taken: 
        """
        if not self.logger:
            return

        # 
        from utils.openai_utils import get_response_text

        # 
        response_text = get_response_text(response)

        # 
        response_record = {
            "frame_number": frame_number,
            "timestamp": time.time(),  # 
            "action_taken": action_taken,
            "response_text": response_text,
            "raw_response": str(response) if response else ""
        }

        # 
        self.model_responses.append(response_record)

        # 
        self.logger.log_info(f" Model Response - Frame {frame_number}:")
        self.logger.log_info(f"   Action: {action_taken}")
        rsp = str(response_text)
        self.logger.log_info(f"   Response: {rsp}...")
        # self.logger.log_info(f"   Response: {response_text[:200]}...")

    def execute_action(self, action: Action) -> bool:
        """
        

        Args:
            action: 

        Returns:
            
        """
        if not self.game:
            self.logger.log_error(" Game not initialized for action execution")
            return False

        try:
            action_type = action.metadata.get("action_type", "unknown")
            face = action.metadata.get("face", "")
            direction = action.metadata.get("direction", "")

            # 
            if action_type == "face_rotation":
                #  (90)
                result = self.game.rotate_face(face, direction)
                if result:
                    self.logger.log_info(f" Face rotation executed: {face} {direction}")
                    # 
                    self._wait_for_animation()
                    return True
                else:
                    self.logger.log_error(f" Face rotation failed: {face} {direction}")
                    return False

            elif action_type == "view_rotation":
                #  (45)
                result = self.game.rotate_cube(face, direction)
                if result:
                    self.logger.log_info(f" View rotation executed: {face} {direction}")
                    # 
                    return True
                else:
                    self.logger.log_error(f" View rotation failed: {face} {direction}")
                    return False

            else:
                self.logger.log_warning(f" Unknown action type: {action_type}")
                return False

        except Exception as e:
            self.logger.log_error(f" Action execution failed: {e}", exc_info=True)
            return False

    def _wait_for_animation(self, max_wait_time: float = 5.0):
        """
        

        Args:
            max_wait_time: 
        """
        if not self.game:
            return

        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # 
            self.game.update()
            # 
            self.game.render()
            state = self.game.get_state()
            if not state.get("animation_in_progress", False):
                break
            # CPU
            time.sleep(0.05)

        if time.time() - start_time >= max_wait_time:
            self.logger.log_warning(f" Animation wait timeout after {max_wait_time}s")

    async def openai_ai_model(self, frame_data: FrameData) -> Action:
        """
        OpenAI AI - MLLM

        Args:
            frame_data: 

        Returns:
            Action: AI
        """
        self.logger.log_info(f" OpenAI AI Model called for frame {frame_data.frame_number}")

        try:
            # 
            screenshot = self.capture_game_screenshot()

            # 
            is_first_call = not self.conversation_history
            full_prompt = self.build_full_prompt(is_first_call)

            conversation_history = build_conversation_history(
                screenshot=screenshot,
                prompt_text=full_prompt,
                step_number=frame_data.frame_number + 1,
                is_first_call=is_first_call,
                previous_history=self.conversation_history,
                history_window=self.history_window
            )

            # OpenAI API
            response = create_openai_response(
                client=self.client,
                model=self.model,
                conversation_history=conversation_history,
                retry_times=self.retry_times
            )

            # build_conversation_history
            self.conversation_history = conversation_history

            # AI
            from utils.openai_utils import get_response_text
            response_text = get_response_text(response)
            self.conversation_history.append({
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": response_text}
                ]
            })

            # action_parser
            action_name, action_data = extract_action_info_from_response(response)

            # 
            action = None
            if action_name and action_data:
                # 
                if "function_name" in action_data:
                    # 
                    function_name = action_data["function_name"]

                    if function_name in RUBIK_ACTIONS:
                        face, direction = RUBIK_ACTIONS[function_name]

                        # 
                        if function_name.startswith("rotate_cube"):
                            #  (45)
                            key_mapping = {
                                "rotate_cube_x_cw": "x", "rotate_cube_x_acw": "x",
                                "rotate_cube_y_cw": "y", "rotate_cube_y_acw": "y",
                                "rotate_cube_z_cw": "z", "rotate_cube_z_acw": "z"
                            }
                            action = Action(
                                type="key_press",
                                key=key_mapping[function_name],
                                metadata={
                                    "frame_number": frame_data.frame_number,
                                    "ai_model": "openai_ai",
                                    "function_called": function_name,
                                    "response_text": response_text,
                                    "face": face,
                                    "direction": direction,
                                    "action_type": "view_rotation"
                                }
                            )
                        else:
                            #  (90)
                            key_mapping = {
                                "rotate_front_cw": "f", "rotate_front_acw": "f",
                                "rotate_back_cw": "b", "rotate_back_acw": "b",
                                "rotate_left_cw": "l", "rotate_left_acw": "l",
                                "rotate_right_cw": "r", "rotate_right_acw": "r",
                                "rotate_up_cw": "u", "rotate_up_acw": "u",
                                "rotate_down_cw": "d", "rotate_down_acw": "d"
                            }
                            action = Action(
                                type="key_press",
                                key=key_mapping[function_name],
                                metadata={
                                    "frame_number": frame_data.frame_number,
                                    "ai_model": "openai_ai",
                                    "function_called": function_name,
                                    "response_text": response_text,
                                    "face": face,
                                    "direction": direction,
                                    "action_type": "face_rotation"
                                }
                            )
                elif "json_data" in action_data:
                    # JSON
                    json_data = action_data["json_data"]
                    parsed_action_name, face, direction = parse_action_from_json_rubic(json_data, RUBIK_ACTIONS)

                    if parsed_action_name and face and direction:
                        # 
                        if parsed_action_name.startswith("rotate_cube"):
                            action_type = "view_rotation"
                            key_mapping = {
                                "rotate_cube_x_cw": "x", "rotate_cube_x_acw": "x",
                                "rotate_cube_y_cw": "y", "rotate_cube_y_acw": "y",
                                "rotate_cube_z_cw": "z", "rotate_cube_z_acw": "z"
                            }
                        else:
                            action_type = "face_rotation"
                            key_mapping = {
                                "rotate_front_cw": "f", "rotate_front_acw": "f",
                                "rotate_back_cw": "b", "rotate_back_acw": "b",
                                "rotate_left_cw": "l", "rotate_left_acw": "l",
                                "rotate_right_cw": "r", "rotate_right_acw": "r",
                                "rotate_up_cw": "u", "rotate_up_acw": "u",
                                "rotate_down_cw": "d", "rotate_down_acw": "d"
                            }

                        key = key_mapping.get(parsed_action_name, "f")

                        action = Action(
                            type="key_press",
                            key=key,
                            metadata={
                                "frame_number": frame_data.frame_number,
                                "ai_model": "openai_ai",
                                "function_called": parsed_action_name,
                                "response_text": response_text,
                                "face": face,
                                "direction": direction,
                                "action_type": action_type,
                                "parsed_from_json": True
                            }
                        )

                # AI
                if action:
                    self.logger.log_info(f" AI Decision: {action_name} - Frame: {frame_data.frame_number}")
                    # 
                    self.record_model_response(frame_data.frame_number, response, action_name)

            # 
            if not action:
                action = Action(
                    type="key_press",
                    key="f",  # 
                    metadata={
                        "frame_number": frame_data.frame_number,
                        "ai_model": "openai_ai",
                        "function_called": "default_rotate",
                        "response_text": response_text,
                        "action_type": "face_rotation",
                        "face": "F",
                        "direction": "CW"
                    }
                )
                self.logger.log_warning(f" No valid action found, using default action")
                # 
                self.record_model_response(frame_data.frame_number, response, "default_rotate")

            # 
            if action:
                success = self.execute_action(action)
                if not success:
                    self.logger.log_error(f" Action execution failed for frame {frame_data.frame_number}")

            return action

        except Exception as e:
            self.logger.log_error(f" OpenAI API all retries exhausted: {e}", exc_info=True)
            # API
            self.api_fatal_failure = True
            if self.engine:
                self.engine.stop_evaluation()
            return Action(
                type="key_press",
                key="f",
                metadata={
                    "frame_number": frame_data.frame_number,
                    "ai_model": "openai_ai",
                    "error": str(e),
                    "note": "API fatal failure - evaluation aborted"
                }
            )

    def frame_callback(self, frame_data: FrameData):
        """ - """
        # 50
        if frame_data.frame_number % 50 == 0:
            self.logger.log_info(f" Frame {frame_data.frame_number} captured")

    def state_callback(self, game_state):
        """ - """
        state = game_state.normalized_state

        # 
        if state.get("cube_solved"):
            self.logger.log_info(f" RUBIK'S CUBE SOLVED! Moves: {state.get('moves_count', 0)}")
            self.logger.log_info(f"   Time: {state.get('time_elapsed', 0):.1f}s")
            # 
            self.game_completed = True
            # 
            if self.engine:
                self.engine.stop_evaluation()

        # 10
        if state.get("moves_count", 0) % 10 == 0 and state.get("moves_count", 0) > 0:
            self.logger.log_info(f" Progress: Move {state.get('moves_count', 0)}, "
                               f"Time: {state.get('time_elapsed', 0):.1f}s, "
                               f"Animation: {state.get('animation_in_progress', False)}")

        # 
        if hasattr(self, '_last_animation_state'):
            current_animation = state.get("animation_in_progress", False)
            if current_animation != self._last_animation_state:
                if current_animation:
                    self.logger.log_info(f" Animation started")
                else:
                    self.logger.log_info(f" Animation completed")
        self._last_animation_state = state.get("animation_in_progress", False)

    def action_callback(self, action: Action):
        """ - AI"""
        frame_number = action.metadata.get("frame_number", 0)
        function_called = action.metadata.get("function_called", "unknown")
        action_type = action.metadata.get("action_type", "unknown")
        face = action.metadata.get("face", "")
        direction = action.metadata.get("direction", "")

        # 10
        if frame_number % 10 == 0:
            self.logger.log_info(f" AI Action - Frame {frame_number}: {function_called}")
            self.logger.log_info(f"   Type: {action_type}, Face: {face}, Direction: {direction}")

        # 
        if action_type in ["face_rotation", "view_rotation"]:
            self.logger.log_info(f" Action executed - Frame {frame_number}: {face} {direction}")

    def save_model_responses(self):
        """JSON"""
        if not self.model_responses:
            self.logger.log_info(" No model responses to save")
            return

        try:
            # 
            os.makedirs(self.log_dir, exist_ok=True)

            # 
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_dir}/rubik_model_responses_{self.config_name}_{timestamp}.json"

            # 
            response_data = {
                "evaluation_info": {
                    "model": self.model,
                    "total_responses": len(self.model_responses),
                    "timestamp": timestamp,
                    "successful_actions": sum(1 for r in self.model_responses if r["action_taken"] != "default_rotate"),
                    "default_actions": sum(1 for r in self.model_responses if r["action_taken"] == "default_rotate")
                },
                "responses": self.model_responses
            }

            # 
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

            self.logger.log_info(f" Model responses saved to: {filename}")
            self.logger.log_info(f"   Total responses: {len(self.model_responses)}")
            self.logger.log_info(f"   Successful actions: {response_data['evaluation_info']['successful_actions']}")
            self.logger.log_info(f"   Default actions: {response_data['evaluation_info']['default_actions']}")

        except Exception as e:
            self.logger.log_error(f" Failed to save model responses: {e}")

    async def run_evaluation(self, max_steps: int = 100, initial_prompt: str = None, config_name: str = "simple", log_dir: str = "logs") -> Dict[str, Any]:
        """
         - MLLM

        Args:
            max_steps: 
            initial_prompt: 
            config_name: 

        Returns:
            
        """
        # 
        self.game_completed = False

        # 
        self.log_dir = log_dir
        self.config_name = config_name

        # 
        self.logger = setup_global_logger(
            name="rubik_openai_evaluation",
            log_level=logging.INFO,
            log_to_file=True,
            log_dir=log_dir
        )

        self.logger.log_info(" Starting Rubik's Cube OpenAI API Evaluation")
        self.logger.log_info(f"   AI Model: OpenAI {self.model}")
        self.logger.log_info(f"   Max steps: {max_steps}")
        self.logger.log_info(f"   Config: {config_name}")
        self.logger.log_info(f"   Headless mode: True")
        self.logger.log_info(f"   Video recording: Enabled")

        # 
        if initial_prompt is None:
            self.current_prompt = SYSTEM_PROMPT
        else:
            self.current_prompt = initial_prompt

        self.logger.log_info(f"   Prompt type: {'custom' if initial_prompt else 'default'}")

        # 
        self.game = RubikCubeGame(config_name=config_name)

        # Pygame
        self.input_source = PygameInputSource()

        # 
        success = self.input_source.initialize(
            game_module=self.game,
            headless=True,
            screen_size=(500, 500)
        )

        if not success:
            self.logger.log_error(" Failed to initialize input source for Rubik's Cube")
            return None

        self.logger.log_input_source_init(self.input_source.get_info())

        # 
        self.engine = EvaluationEngine(
            input_source=self.input_source,
            decision_frequency=0.5,  # 2/OpenAI API
            record_video=True,       # 
            video_fps=0.5,           # 
            video_output_dir="videos"
        )

        # 
        self.engine.add_frame_callback(self.frame_callback)
        self.engine.add_state_callback(self.state_callback)
        self.engine.add_action_callback(self.action_callback)

        self.logger.log_evaluation_start(max_steps=max_steps, max_duration=None)

        # 
        try:
            stats = await self.engine.run_evaluation(
                ai_model_func=self.openai_ai_model,
                max_steps=max_steps,
                max_duration=None,
                session_name="rubik_openai_evaluation"
            )

            # 
            self.logger.log_evaluation_end(stats)
            self.logger.log_performance_metrics(stats)

            # 
            final_state = self.input_source.get_game_state()
            if final_state:
                state = final_state.normalized_state
                self.logger.log_info("\n=== FINAL GAME RESULTS ===")
                self.logger.log_info(f"Cube Solved: {state.get('cube_solved', False)}")
                self.logger.log_info(f"Steps taken: {stats['total_frames']}")
                self.logger.log_info(f"Time elapsed: {state.get('time_elapsed', 0):.1f}s")
                self.logger.log_info(f"Config used: {state.get('config_name', 'unknown')}")

            # 
            if os.path.exists("videos"):
                video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
                if video_files:
                    self.logger.log_info(f" Video files recorded: {len(video_files)}")

            if self.api_fatal_failure:
                self.logger.log_error(" Evaluation aborted: API all retries exhausted, skipping level")
                return None

            return {
                "cube_solved": state.get('cube_solved', False) if final_state else False,
                "steps": stats['total_frames'],
                "max_steps": max_steps,
                "time_elapsed": state.get('time_elapsed', 0) if final_state else 0,
                "config_name": config_name,
                "stats": stats
            }

        except Exception as e:
            self.logger.log_error(f" Rubik's Cube evaluation failed: {e}", exc_info=True)
            return None
        finally:
            # 
            self.save_model_responses()

            # 
            self.input_source.close()
            self.logger.log_info(" Rubik's Cube evaluation completed")


def parse_arguments():
    """"""
    parser = argparse.ArgumentParser(description='OpenAI API')
    parser.add_argument('--max-steps', type=int, default=EVALUATION_CONFIG.get('max_steps', 100),
                       help=f' (: {EVALUATION_CONFIG.get("max_steps", 100)})')
    parser.add_argument('--custom-prompt', type=str,
                       help=' ()')
    parser.add_argument('--level',
                       default=GAME_CONFIG["initial_config"], help=' (: 1)')
    parser.add_argument('--api-base-url', type=str, default=OPENAI_CONFIG['api_base_url'],
                       help='OpenAI APIURL')
    parser.add_argument('--api-key', type=str, default=OPENAI_CONFIG['api_key'],
                       help='OpenAI API')
    parser.add_argument('--model', type=str, default=OPENAI_CONFIG['model'],
                       help='')
    parser.add_argument('--log-dir', type=str, default=OUTPUT_CONFIG.get('log_dir', 'logs'),
                       help=f' (: {OUTPUT_CONFIG.get("log_dir", "logs")})')
    parser.add_argument('--retry-times', type=int, default=EVALUATION_CONFIG.get('retry_times', 3),
                       help=f'API (: {EVALUATION_CONFIG.get("retry_times", 3)})')

    return parser.parse_args()


async def main():
    """ - """
    args = parse_arguments()

    print("=" * 60)
    print("OpenAI API (MLLM)")
    print("=" * 60)
    print(f":")
    print(f"  : {args.max_steps}")
    print(f"  : {args.level}")
    print(f"  API: {args.model}")
    print("=" * 60)

    # 
    evaluator = RubikOpenAIEvaluator(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model=args.model
    )
    evaluator.retry_times = args.retry_times

    # 
    if args.custom_prompt:
        initial_prompt = args.custom_prompt
        print("")
    else:
        initial_prompt = SYSTEM_PROMPT
        print("")

    # 
    result = await evaluator.run_evaluation(
        max_steps=args.max_steps,
        initial_prompt=initial_prompt,
        config_name=args.level,
        log_dir=args.log_dir
    )

    # 
    if result:
        print("\n" + "="*60)
        print(":")
        print(f": {'' if result['cube_solved'] else ''}")
        print(f": {result['steps']}/{result['max_steps']}")
        print(f": {result['time_elapsed']:.1f}")
        print(f": {result['config_name']}")

        # 
        stats = result['stats']
        print(f"\n :")
        print(f"  : {stats['total_frames']}")
        print(f"  : {stats['total_time']:.2f}s")
        print(f"  FPS: {stats['actual_fps']:.2f}")
        print(f"  : {stats['avg_inference_time']:.3f}s")

        # 
        logger = get_logger()
        log_file = logger.get_log_file_path()
        if log_file:
            print(f"\n : {log_file}")

        if os.path.exists("videos"):
            video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
            if video_files:
                print(f" :")
                for video_file in video_files:
                    print(f"   - videos/{video_file}")

        print("="*60)

        # 
        summary = {
            "cube_solved": result["cube_solved"],
            "steps": result["steps"],
            "max_steps": result["max_steps"],
            "time_elapsed": result["time_elapsed"],
            "config_name": result["config_name"],
            "model": args.model,
            "total_time": stats['total_time'],
            "total_frames": stats['total_frames']
        }

        # 
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"{args.log_dir}/rubik_evaluation_summary_{args.level}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f": {args.log_dir}/rubik_evaluation_summary_{args.level}_{timestamp}.json")
    else:
        print("\n API")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
