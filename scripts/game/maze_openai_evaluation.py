#!/usr/bin/env python3
"""
OpenAI API (2)
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
from games.maze3d.maze3d_adapter import Maze3DGame
from input_sources.pygame_input_source import PygameInputSource
from core.evaluation_engine import EvaluationEngine
from core.data_classes import FrameData, Action
from utils.logger import setup_global_logger, get_logger
from utils.openai_utils import encode_image, create_openai_response, build_conversation_history, add_response_to_history, get_response_text
from utils.action_parser import extract_action_info_from_response

# 
from maze_config import OPENAI_CONFIG, SYSTEM_PROMPT, OUTPUT_CONFIG, get_level_file, get_config, GAME_CONFIG, EVALUATION_CONFIG

# 
MAZE_KEY_MAPPING = {
    "move_forward": "w",      # 
    "turn_left": "a",         # 
    "turn_right": "d"         # 
}

MOVEMENT_ACTIONS = {"move_forward"}
GRANULARITY_STEPS = {
    "small": 1,
    "medium": 2,
    "large": 3,
}


def get_movement_granularity(action_name: str, action_data: Dict[str, Any]) -> tuple[str, int]:
    """ small/1"""
    if action_name not in MOVEMENT_ACTIONS:
        return "small", 1

    granularity = action_data.get("granularity", "small")
    if granularity is None:
        granularity = "small"
    granularity = str(granularity).strip().lower()
    if granularity == "midium":
        granularity = "medium"

    if granularity not in GRANULARITY_STEPS:
        granularity = "small"

    return granularity, GRANULARITY_STEPS[granularity]


class MazeOpenAIEvaluator:
    """OpenAI - MLLM"""

    def __init__(self, api_base_url: str, api_key: str, model: str = "gpt-5", maze_file: str = None, level_number: int = None):
        """
        

        Args:
            api_base_url: OpenAI APIURL
            api_key: API
            model: 
            maze_file: level_number
            level_number:  (1-20)
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

        # 
        if maze_file:
            self.maze_file = maze_file
            self.level_number = None
        elif level_number:
            self.maze_file = get_level_file(level_number)
            self.level_number = level_number
        else:
            # 
            config = get_config()
            self.maze_file = config["game"]["maze_file"]
            self.level_number = config["game"]["level_number"]

        # 
        self.tools = [
            {
                "type": "function",
                "name": "move_forward",
                "description": "Move forward with a granularity parameter: small=1 grid cell, medium=2 grid cells, large=3 grid cells. Movement stops early if blocked by a wall.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "granularity": {
                            "type": "string",
                            "enum": ["small", "medium", "large"]
                        }
                    },
                    "required": ["granularity"],
                },
            },
            {
                "type": "function",
                "name": "turn_left",
                "description": "Turn left 90 degrees",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "turn_right",
                "description": "Turn right 90 degrees",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            # {
            #     "type": "function",
            #     "name": "get_game_state",
            #     "description": "Get current game state including player position, direction, steps taken, and other information",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {},
            #         "required": [],
            #     },
            # },
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
You must respond with a JSON object containing the action you want to take.

For movement actions, use EXACTLY this format:
{
  "action": "function_name",
  "granularity": "small|medium|large"
}

For turn actions, use EXACTLY this format:
{
  "action": "function_name"
}

Important rules:
1. Movement actions MUST contain ONLY the "action" and "granularity" fields
2. Turn actions MUST contain ONLY the "action" field
3. For movement actions, "granularity" controls distance: "small" = 1 grid cell, "medium" = 2 grid cells, "large" = 3 grid cells
4. Do NOT include any other fields like "face", "direction", "key", etc.
5. Do NOT include any additional text or explanations

Valid examples:
- To move forward 1 grid cell: {"action": "move_forward", "granularity": "small"}
- To move forward 2 grid cells: {"action": "move_forward", "granularity": "medium"}
- To move forward 3 grid cells: {"action": "move_forward", "granularity": "large"}
- To turn left: {"action": "turn_left"}
- To turn right: {"action": "turn_right"}

Invalid examples (DO NOT USE):
- {"action": "move_forward"}  # WRONG: missing granularity
- {"action": "move_forward", "granularity": "midium"}  # WRONG: use "medium" (deprecated typo)
- {"action": "turn_left", "granularity": "small"}  # WRONG: turn actions do not take granularity
- {"action": "move_forward", "face": "E", "direction": "CW"}  # WRONG: contains extra fields
- {"action": "turn_left", "key": "a"}  # WRONG: contains extra field
- I want to move forward: {"action": "move_forward", "granularity": "small"}  # WRONG: contains extra text

IMPORTANT: Respond with ONLY the JSON object. Add "granularity" only for movement actions.
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
        response_record = {
            "frame_number": frame_number,
            "timestamp": time.time(),  # 
            "action_taken": action_taken,
            "response_text": get_response_text(response),
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
        self.logger.log_info(f"   Response: {str(response_record['response_text'])[:200]}...")
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
                    key = MAZE_KEY_MAPPING.get(function_name)

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
                        key = MAZE_KEY_MAPPING.get(action_name, "w")

                # 
                if key:
                    # action_data
                    direction = action_data.get("direction", "forward")
                    action_type = action_data.get("type", "move")
                    granularity, repeat = get_movement_granularity(action_name, action_data)

                    metadata = {
                        "frame_number": frame_data.frame_number,
                        "ai_model": "openai_ai",
                        "function_called": action_name,
                        "response_text": response_text,
                        "direction": direction,
                        "action_type": action_type,
                        "parsed_from_json": "json_data" in action_data,
                        "parsed_from_function_call": "function_name" in action_data
                    }
                    if action_name in MOVEMENT_ACTIONS:
                        metadata["granularity"] = granularity
                        metadata["repeat"] = repeat

                    action = Action(
                        type="key_press",
                        key=key,
                        metadata=metadata
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
                    key="w",  # 
                    metadata={
                        "frame_number": frame_data.frame_number,
                        "ai_model": "openai_ai",
                        "function_called": "default_move",
                        "response_text": response_text,
                        "granularity": "small",
                        "repeat": 1,
                        "note": "No valid action found, using default action"
                    }
                )
                self.logger.log_warning(f" No valid action found, using default action")
                # 
                self.record_model_response(frame_data.frame_number, response, "default_move")

            return action

        except Exception as e:
            self.logger.log_error(f" OpenAI API all retries exhausted: {e}", exc_info=True)
            # API
            self.api_fatal_failure = True
            if self.engine:
                self.engine.stop_evaluation()
            return Action(
                type="key_press",
                key="w",
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
        if state.get("game_won"):
            self.logger.log_info(f" MAZE COMPLETED! Steps: {state.get('steps_taken', 0)}")
            self.logger.log_info(f"   Distance to exit: {state.get('distance_to_exit', 0)}")
            # 
            self.game_completed = True
            # 
            if self.engine:
                self.engine.stop_evaluation()

        elif state.get("game_over") and not state.get("game_won"):
            self.logger.log_info(f" GAME OVER - Failed to complete maze")
            self.logger.log_info(f"   Steps taken: {state.get('steps_taken', 0)}")
            self.logger.log_info(f"   Distance to exit: {state.get('distance_to_exit', 0)}")
            # 
            self.game_completed = True
            # 
            if self.engine:
                self.engine.stop_evaluation()

        # 50
        if state.get("steps_taken", 0) % 50 == 0 and state.get("steps_taken", 0) > 0:
            self.logger.log_info(f" Progress: Step {state.get('steps_taken', 0)}, "
                               f"Distance: {state.get('distance_to_exit', 0)}")

    def action_callback(self, action: Action):
        """ - AI"""
        frame_number = action.metadata.get("frame_number", 0)
        function_called = action.metadata.get("function_called", "unknown")

        # 20
        if frame_number % 20 == 0:
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
            level_str = self.level_number if self.level_number else "default"
            filename = f"{self.log_dir}/maze_model_responses_level{level_str}_{timestamp}.json"

            # 
            response_data = {
                "evaluation_info": {
                    "model": self.model,
                    "total_responses": len(self.model_responses),
                    "timestamp": timestamp,
                    "successful_actions": sum(1 for r in self.model_responses if r["action_taken"] != "default_move"),
                    "default_actions": sum(1 for r in self.model_responses if r["action_taken"] == "default_move")
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

    async def run_evaluation(self, max_steps: int = 50, initial_prompt: str = None, log_dir: str = "logs") -> Dict[str, Any]:
        """
         - MLLM

        Args:
            max_steps: 
            initial_prompt: 

        Returns:
            
        """
        # 
        self.game_completed = False

        # 
        self.log_dir = log_dir

        # 
        self.logger = setup_global_logger(
            name="maze_openai_evaluation",
            log_level=logging.INFO,
            log_to_file=True,
            log_dir=log_dir
        )

        self.logger.log_info(" Starting Maze Game OpenAI API Evaluation")
        self.logger.log_info(f"   AI Model: OpenAI {self.model}")
        self.logger.log_info(f"   Max steps: {max_steps}")
        self.logger.log_info(f"   Headless mode: True")
        self.logger.log_info(f"   Video recording: Enabled")

        # 
        if initial_prompt is None:
            self.current_prompt = SYSTEM_PROMPT
        else:
            self.current_prompt = initial_prompt

        self.logger.log_info(f"   Prompt type: {'custom' if initial_prompt else 'default'}")

        # 3D
        self.game = Maze3DGame(maze_file=self.maze_file)

        # 
        self.logger.log_info(f"   Maze file: {self.maze_file}")
        if hasattr(self, 'level_number'):
            self.logger.log_info(f"   Level number: {self.level_number}")

        # Pygame
        self.input_source = PygameInputSource()

        # 
        success = self.input_source.initialize(
            game_module=self.game,
            headless=True,
            screen_size=(800, 600)
        )

        if not success:
            self.logger.log_error(" Failed to initialize input source for 3D maze")
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
                session_name="maze_openai_evaluation"
            )

            # 
            self.logger.log_evaluation_end(stats)
            self.logger.log_performance_metrics(stats)

            # 
            final_state = self.input_source.get_game_state()
            if final_state:
                state = final_state.normalized_state
                self.logger.log_info("\n=== FINAL GAME RESULTS ===")
                self.logger.log_info(f"Success: {state.get('success', False)}")
                self.logger.log_info(f"Steps taken: {state.get('steps_taken', 0)}")
                self.logger.log_info(f"Distance to exit: {state.get('distance_to_exit', 0)}")
                self.logger.log_info(f"Player direction: {state.get('player_direction', 'UNKNOWN')}")

            # 
            if os.path.exists("videos"):
                video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
                if video_files:
                    self.logger.log_info(f" Video files recorded: {len(video_files)}")

            if self.api_fatal_failure:
                self.logger.log_error(" Evaluation aborted: API all retries exhausted, skipping level")
                return None

            return {
                "success": state.get('success', False) if final_state else False,
                "steps_taken": state.get('steps_taken', 0) if final_state else 0,
                "max_steps": max_steps,
                "distance_to_exit": state.get('distance_to_exit', 0) if final_state else 0,
                "stats": stats
            }

        except Exception as e:
            self.logger.log_error(f" Maze evaluation failed: {e}", exc_info=True)
            return None
        finally:
            # 
            self.save_model_responses()

            # 
            self.input_source.close()
            self.logger.log_info(" Maze evaluation completed")


def parse_arguments():   
    """"""
    parser = argparse.ArgumentParser(description='OpenAI API')
    parser.add_argument('--max-steps', type=int, default=EVALUATION_CONFIG['max_steps'],
                       help=f' (: {EVALUATION_CONFIG["max_steps"]})')
    parser.add_argument('--custom-prompt', type=str,
                       help=' ()')
    parser.add_argument('--api-base-url', type=str, default=OPENAI_CONFIG['api_base_url'],
                       help='OpenAI APIURL')
    parser.add_argument('--api-key', type=str, default=OPENAI_CONFIG['api_key'],
                       help='OpenAI API')
    parser.add_argument('--model', type=str, default=OPENAI_CONFIG['model'],
                       help='')
    parser.add_argument('--level', type=int, default=GAME_CONFIG["level_number"],
                       help=' (1-20)')
    parser.add_argument('--maze-file', type=str, default=None,
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
    print(f"  API: {args.model}")
    if args.level:
        print(f"  : {args.level}")
    elif args.maze_file:
        print(f"  : {args.maze_file}")
    else:
        print(f"  : Level 01")
    print("=" * 60)

    # 
    evaluator = MazeOpenAIEvaluator(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model=args.model,
        maze_file=args.maze_file,
        level_number=args.level
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
        log_dir=args.log_dir
    )

    # 
    if result:
        print("\n" + "="*60)
        print(":")
        print(f": {'' if result['success'] else ''}")
        print(f": {result['steps_taken']}/{result['max_steps']}")
        print(f": {result['distance_to_exit']} ")

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
            "success": result["success"],
            "steps_taken": result["steps_taken"],
            "max_steps": result["max_steps"],
            "distance_to_exit": result["distance_to_exit"],
            "model": args.model,
            "total_time": stats['total_time'],
            "total_frames": stats['total_frames'],
            "maze_file": evaluator.maze_file,
            "level_number": evaluator.level_number
        }

        # 
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        level_str = evaluator.level_number if evaluator.level_number else "default"

        with open(f"{args.log_dir}/maze_evaluation_summary_level{level_str}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f": {args.log_dir}/maze_evaluation_summary_level{level_str}_{timestamp}.json")
    else:
        print("\n API")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
