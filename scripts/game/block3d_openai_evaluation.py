#!/usr/bin/env python3
"""
3DOpenAI API
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
from games.Block3D.block3d_adapter import Block3DGameAdapter
from input_sources.pygame_input_source import PygameInputSource
from core.evaluation_engine import EvaluationEngine
from core.data_classes import FrameData, Action
from utils.logger import setup_global_logger, get_logger
from utils.openai_utils import encode_image, create_openai_response, build_conversation_history, add_response_to_history
from utils.action_parser import extract_action_info_from_response

# 
from block3d_config import OPENAI_CONFIG, SYSTEM_PROMPT, GAME_CONFIG, EVALUATION_CONFIG, OUTPUT_CONFIG

# 3D
BLOCK3D_KEY_MAPPING = {
    "move_left": "left",        # X
    "move_right": "right",      # X
    "move_up": "up",            # Y
    "move_down": "down",        # Y
    "move_forward": "w",        # Z
    "move_backward": "s",       # Z
    "place_block": "space"      # 
}


class Block3DOpenAIEvaluator:
    """3DOpenAI - MLLM"""

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
        self.level = 0  # 

        #  - 7
        self.tools = [
            {
                "type": "function",
                "name": "move_left",
                "description": "Move cursor left (-X direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_right",
                "description": "Move cursor right (+X direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_up",
                "description": "Move cursor up (-Y direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_down",
                "description": "Move cursor down (+Y direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_forward",
                "description": "Move cursor forward (+Z direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_backward",
                "description": "Move cursor backward (-Z direction). Cannot move beyond grid boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "place_block",
                "description": "Place or remove a block at cursor position. Toggles block presence.",
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
You must respond with a JSON object containing the action you want to take. The JSON should have EXACTLY the following format:
{
  "action": "function_name"
}

Important rules:
1. The JSON must contain ONLY the "action" field
2. Do NOT include any other fields like "face", "direction", "key", etc.
3. Do NOT include any additional text or explanations

Valid examples:
- To move left: {"action": "move_left"}
- To move right: {"action": "move_right"}
- To move up: {"action": "move_up"}
- To move down: {"action": "move_down"}
- To move forward: {"action": "move_forward"}
- To move backward: {"action": "move_backward"}
- To place block: {"action": "place_block"}

Invalid examples (DO NOT USE):
- {"action": "move_left", "direction": "left"}  # WRONG: contains extra field
- {"action": "place_block", "key": "space"}  # WRONG: contains extra field
- I want to move left: {"action": "move_left"}  # WRONG: contains extra text

IMPORTANT: You must respond with ONLY the JSON object containing EXACTLY the "action" field, no other fields, no additional text.
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
            "function_calls": [],
            "raw_response": str(response) if response else ""
        }

        # 
        from utils.openai_utils import extract_function_calls
        function_calls = extract_function_calls(response)
        for func_call in function_calls:
            response_record["function_calls"].append({
                "function_name": func_call.get("function_name", ""),
                "call_id": func_call.get("call_id", "N/A"),
                "arguments": func_call.get("arguments", {})
            })

        # 
        self.model_responses.append(response_record)

        # 
        self.logger.log_info(f" Model Response - Frame {frame_number}:")
        self.logger.log_info(f"   Action: {action_taken}")
        rsp = str(response_text)
        self.logger.log_info(f"   Response: {rsp}...")
        if response_record['function_calls']:
            self.logger.log_info(f"   Function calls: {[fc['function_name'] for fc in response_record['function_calls']]}")

    async def openai_ai_model(self, frame_data: FrameData) -> Action:
        """
        OpenAI AI - MLLM

        Args:
            frame_data: 

        Returns:
            Action: AI
        """
        self.logger.log_info(f" OpenAI AI Model called for frame {frame_data.frame_number}")

        # 
        if self.game and hasattr(self.game, 'should_continue'):
            if not self.game.should_continue():
                # 
                game_state = self.game.get_state()
                raise RuntimeError(f"Game over at step {game_state.get('steps_taken', 0)}, "
                                 f"score: {game_state.get('score', 0)}, "
                                 f"progress: {game_state.get('completion_percentage', 0):.1f}%")

        try:
            # 
            screenshot = self.capture_game_screenshot()

            #  - build_conversation_history
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
                key = None

                # 
                if "function_name" in action_data:
                    function_name = action_data["function_name"]
                    action_name = function_name  # 

                    # 
                    key = BLOCK3D_KEY_MAPPING.get(function_name)

                # JSON
                elif "json_data" in action_data:
                    json_data = action_data["json_data"]
                    action_name = json_data.get("action", action_name)

                    # JSONkey
                    key = json_data.get("key")
                    # keyaction_data
                    if not key:
                        key = action_data.get("key")
                    # key
                    if not key and action_name:
                        key = BLOCK3D_KEY_MAPPING.get(action_name, "right")

                # 
                if key:
                    # action_data
                    direction = action_data.get("direction", "forward")
                    action_type = action_data.get("type", "move")

                    action = Action(
                        type="key_press",
                        key=key,
                        metadata={
                            "frame_number": frame_data.frame_number,
                            "ai_model": "openai_ai",
                            "function_called": action_name,
                            "response_text": response_text,
                            "direction": direction,
                            "action_type": action_type,
                            "parsed_from_json": "json_data" in action_data,
                            "parsed_from_function_call": "function_name" in action_data
                        }
                    )

                # AI
                if action:
                    self.logger.log_info(f" AI Decision: {action_name} - Frame: {frame_data.frame_number}")
                    # 
                    self.record_model_response(frame_data.frame_number, response, action_name)

            # 
            if not action:
                # 
                game_state = self.game.get_state()
                cursor_x = game_state.get("cursor_x", 2)
                cursor_y = game_state.get("cursor_y", 2)
                cursor_z = game_state.get("cursor_z", 0)
                matched_blocks = game_state.get("matched_blocks", 0)
                target_blocks = game_state.get("target_blocks", 1)

                # 
                possible_actions = []

                # 
                #  (x-1)
                if cursor_x > 0:
                    possible_actions.append(("move_left", "left"))
                #  (x+1)
                if cursor_x < 5:
                    possible_actions.append(("move_right", "right"))
                #  (y-1)
                if cursor_y > 0:
                    possible_actions.append(("move_up", "up"))
                #  (y+1)
                if cursor_y < 5:
                    possible_actions.append(("move_down", "down"))
                #  (z+1)
                if cursor_z < 5:
                    possible_actions.append(("move_forward", "w"))
                #  (z-1)
                if cursor_z > 0:
                    possible_actions.append(("move_backward", "s"))

                # 
                possible_actions.append(("place_block", "space"))

                # 
                if matched_blocks < target_blocks:
                    # 
                    default_action, default_key = "place_block", "space"
                elif possible_actions:
                    # 
                    import random
                    default_action, default_key = random.choice(possible_actions)
                else:
                    # 
                    default_action, default_key = "move_right", "right"

                action = Action(
                    type="key_press",
                    key=default_key,
                    metadata={
                        "frame_number": frame_data.frame_number,
                        "ai_model": "openai_ai",
                        "function_called": f"default_{default_action}",
                        "response_text": response_text,
                        "note": f"No valid function call, using safe default: {default_action}"
                    }
                )
                self.logger.log_warning(f" No valid action found, using default action")
                # 
                self.record_model_response(frame_data.frame_number, response, f"default_{default_action}")

            return action

        except Exception as e:
            self.logger.log_error(f" OpenAI API all retries exhausted: {e}", exc_info=True)
            # API
            self.api_fatal_failure = True
            if self.engine:
                self.engine.stop_evaluation()
            return Action(
                type="key_press",
                key="right",  # 
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
        if state.get("game_over"):
            if state.get("won"):
                self.logger.log_info(f" GAME WON - Final Score: {state.get('score', 0)}")
                self.logger.log_info(f"   Completion: {state.get('completion_percentage', 0):.1f}%")
                self.logger.log_info(f"   Steps taken: {state.get('steps_taken', 0)}")
                # 
                self.game_completed = True
                # 
                if self.engine:
                    self.engine.stop_evaluation()
            else:
                self.logger.log_info(f" GAME OVER - Timeout")
                self.logger.log_info(f"   Final progress: {state.get('completion_percentage', 0):.1f}%")
                self.logger.log_info(f"   Steps taken: {state.get('steps_taken', 0)}")
                # 
                self.game_completed = True
                # 
                if self.engine:
                    self.engine.stop_evaluation()

        # 
        current_matched = state.get("matched_blocks", 0)
        if current_matched > self.last_matched:
            self.logger.log_info(f" Block placed correctly! Matched: {current_matched}/{state.get('target_blocks', 1)}")
            self.logger.log_info(f"   Progress: {state.get('completion_percentage', 0):.1f}%")
            self.last_matched = current_matched

        # 20
        if state.get("steps_taken", 0) % 20 == 0 and state.get("steps_taken", 0) > 0:
            self.logger.log_info(f" Progress: Step {state.get('steps_taken', 0)}/{state.get('max_steps', 0)}, "
                               f"Matched: {state.get('matched_blocks', 0)}/{state.get('target_blocks', 1)}, "
                               f"Completion: {state.get('completion_percentage', 0):.1f}%")

    def action_callback(self, action: Action):
        """ - AI"""
        frame_number = action.metadata.get("frame_number", 0)
        function_called = action.metadata.get("function_called", "unknown")

        # 10
        if frame_number % 10 == 0:
            self.logger.log_info(f" AI Action - Frame {frame_number}: {function_called}")

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
            filename = f"{self.log_dir}/block3d_model_responses_level{self.level}_{timestamp}.json"

            # 
            response_data = {
                "evaluation_info": {
                    "model": self.model,
                    "total_responses": len(self.model_responses),
                    "timestamp": timestamp,
                    "successful_actions": sum(1 for r in self.model_responses if not r["action_taken"].startswith("default_")),
                    "default_actions": sum(1 for r in self.model_responses if r["action_taken"].startswith("default_"))
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

    async def run_evaluation(self, max_steps: int = 200, initial_prompt: str = None, level: int = 0, log_dir: str = "logs") -> Dict[str, Any]:
        """
         - MLLM

        Args:
            max_steps: 
            initial_prompt: 
            level:  (1-20)

        Returns:
            
        """
        # 
        self.game_completed = False

        # 
        self.log_dir = log_dir
        self.level = level

        # 
        self.logger = setup_global_logger(
            name="block3d_openai_evaluation",
            log_level=logging.DEBUG,  # DEBUG
            log_to_file=True,
            log_dir=log_dir
        )

        self.logger.log_info(" Starting 3D Block Builder Game OpenAI API Evaluation")
        self.logger.log_info(f"   AI Model: OpenAI {self.model}")
        self.logger.log_info(f"   Max steps: {max_steps}")
        self.logger.log_info(f"   Level: {level} ({self._get_level_name(level)})")
        self.logger.log_info(f"   Grid size: 6x6x6")
        self.logger.log_info(f"   Headless mode: True")
        self.logger.log_info(f"   Video recording: Enabled")

        # 
        if initial_prompt is None:
            self.current_prompt = SYSTEM_PROMPT
        else:
            self.current_prompt = initial_prompt

        self.logger.log_info(f"   Prompt type: {'custom' if initial_prompt else 'default'}")

        # 
        self.game = Block3DGameAdapter(level=level, max_steps=max_steps)

        # Pygame
        self.input_source = PygameInputSource()

        # 
        success = self.input_source.initialize(
            game_module=self.game,
            headless=True,
            screen_size=(self.game.SCREEN_WIDTH, self.game.SCREEN_HEIGHT)
        )

        if not success:
            self.logger.log_error(" Failed to initialize input source for 3D block builder")
            return None

        self.logger.log_input_source_init(self.input_source.get_info())

        # 
        self.engine = EvaluationEngine(
            input_source=self.input_source,
            decision_frequency=1.0,  # 1/OpenAI API
            record_video=True,       # 
            video_fps=1.0,           # 
            video_output_dir="videos"
        )

        # 
        self.engine.add_frame_callback(self.frame_callback)
        self.engine.add_state_callback(self.state_callback)
        self.engine.add_action_callback(self.action_callback)

        self.logger.log_evaluation_start(max_steps=max_steps, max_duration=None)

        # 
        self.last_matched = 0

        # 
        try:
            stats = await self.engine.run_evaluation(
                ai_model_func=self.openai_ai_model,
                max_steps=max_steps,
                max_duration=None,
                session_name="block3d_openai_evaluation"
            )

            # 
            self.logger.log_evaluation_end(stats)
            self.logger.log_performance_metrics(stats)

            # 
            final_state = self.input_source.get_game_state()
            if final_state:
                state = final_state.normalized_state
                self.logger.log_info("\n=== FINAL GAME RESULTS ===")
                self.logger.log_info(f"Final Score: {state.get('score', 0)}")
                self.logger.log_info(f"Matched blocks: {state.get('matched_blocks', 0)}/{state.get('target_blocks', 1)}")
                self.logger.log_info(f"Completion: {state.get('completion_percentage', 0):.1f}%")
                self.logger.log_info(f"Steps taken: {state.get('steps_taken', 0)}/{state.get('max_steps', 0)}")
                self.logger.log_info(f"Game over: {state.get('game_over', False)}")
                self.logger.log_info(f"Success (won): {state.get('won', False)}")

            # 
            if os.path.exists("videos"):
                video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
                if video_files:
                    self.logger.log_info(f" Video files recorded: {len(video_files)}")

            if self.api_fatal_failure:
                self.logger.log_error(" Evaluation aborted: API all retries exhausted, skipping level")
                return None

            return {
                "score": state.get('score', 0) if final_state else 0,
                "matched_blocks": state.get('matched_blocks', 0) if final_state else 0,
                "target_blocks": state.get('target_blocks', 1) if final_state else 1,
                "completion_percentage": state.get('completion_percentage', 0) if final_state else 0,
                "steps_taken": state.get('steps_taken', 0) if final_state else 0,
                "max_steps": max_steps,
                "game_over": state.get('game_over', False) if final_state else False,
                "won": state.get('won', False) if final_state else False,
                "level": level,
                "level_name": self._get_level_name(level),
                "stats": stats
            }

        except Exception as e:
            self.logger.log_error(f" Block3D evaluation failed: {e}", exc_info=True)
            return None
        finally:
            # 
            self.save_model_responses()

            # 
            self.input_source.close()
            self.logger.log_info(" Block3D evaluation completed")

    def _get_level_name(self, level: int) -> str:
        """"""
        level_names = {
            1: " -  (X)",
            8: " - ",
            16: " - ",
            17: " - "
        }
        return level_names.get(level, f" {level}")


def parse_arguments():
    """"""
    parser = argparse.ArgumentParser(description='3DOpenAI API')
    parser.add_argument('--max-steps', type=int, default=EVALUATION_CONFIG['max_steps'],
                       help=f" (: {EVALUATION_CONFIG['max_steps']})")
    parser.add_argument('--level', type=int, default=GAME_CONFIG['level'], choices=list(range(1, 21)),
                       help=' (1-20, : 1)')
    parser.add_argument('--custom-prompt', type=str,
                       help=' ()')
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
    print("3DOpenAI API (MLLM)")
    print("=" * 60)
    print(f":")
    print(f"  : {args.max_steps}")
    print(f"  : {args.level} ({Block3DOpenAIEvaluator._get_level_name(None, args.level)})")
    print(f"  API: {args.model}")
    print(f"  : 6x6x6")
    print("=" * 60)

    # 
    evaluator = Block3DOpenAIEvaluator(
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
        level=args.level,
        log_dir=args.log_dir
    )

    # 
    if result:
        print("\n" + "="*60)
        print(":")
        print(f": {result['score']}")
        print(f": {result['matched_blocks']}/{result['target_blocks']}")
        print(f": {result['completion_percentage']:.1f}%")
        print(f": {result['steps_taken']}/{result['max_steps']}")
        print(f": {'' if result['game_over'] else ''}")
        print(f": {'' if result['won'] else ''}")
        print(f": {result['level']} ({result['level_name']})")

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
            "score": result["score"],
            "matched_blocks": result["matched_blocks"],
            "target_blocks": result["target_blocks"],
            "completion_percentage": result["completion_percentage"],
            "steps_taken": result["steps_taken"],
            "max_steps": result["max_steps"],
            "game_over": result["game_over"],
            "won": result["won"],
            "level": result["level"],
            "level_name": result["level_name"],
            "model": args.model,
            "total_time": stats['total_time'],
            "total_frames": stats['total_frames']
        }

        # 
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"{args.log_dir}/block3d_evaluation_summary_level{args.level}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f": {args.log_dir}/block3d_evaluation_summary_level{args.level}_{timestamp}.json")
    else:
        print("\n API")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
