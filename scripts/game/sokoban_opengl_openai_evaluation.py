#!/usr/bin/env python3
"""
Sokoban Game OpenGL OpenAI API Evaluation Script
Using MLLM framework evaluation engine with complete logging and video recording
3D Sokoban game with OpenGL rendering + OpenAI API
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

# Add project root directory to path
# Add project paths needed when this script is launched from scripts/game.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GAME_ENV_DIR = os.path.join(REPO_ROOT, "mllm_base_agent", "environments", "game")
GAME_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "game")
for path in (REPO_ROOT, GAME_ENV_DIR, GAME_CONFIG_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import Sokoban game and MLLM framework components
from games.SOKOBAN_3D.sokoban_adapter import SokobanGame
from input_sources.pygame_opengl_input_source import PygameOpenGLInputSource
from core.evaluation_engine import EvaluationEngine
from core.data_classes import FrameData, Action
from utils.logger import setup_global_logger, get_logger
from utils.openai_utils import encode_image, create_openai_response, build_conversation_history, add_response_to_history
from utils.action_parser import extract_action_info_from_response

# Import configuration
from sokoban_config import OPENAI_CONFIG, SYSTEM_PROMPT, OUTPUT_CONFIG, LEVEL_CONFIG, EVALUATION_CONFIG

# Sokoban
SOKOBAN_KEY_MAPPING = {
    "move_up": "w",
    "move_down": "s",
    "move_left": "a",
    "move_right": "d"
}


class SokobanOpenGLOpenAIEvaluator:
    """Sokoban Game OpenGL OpenAI Evaluator - Using MLLM Framework"""

    def __init__(self, api_base_url: str, api_key: str, model: str = "gpt-4o"):
        """
        Initialize evaluator

        Args:
            api_base_url: OpenAI API base URL
            api_key: API key
            model: Model name to use
        """
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model = model
        self.logger = None
        self.engine = None
        self.input_source = None
        self.game = None
        self.conversation_history = []  # Store complete conversation history
        self.history_window = EVALUATION_CONFIG.get("history_window", None)  # Sliding history window size
        self.retry_times = EVALUATION_CONFIG.get("retry_times", 3)  # Max API retry times on failure
        self.model_responses = []  # Store each model response
        self.game_completed = False  # Flag indicating whether game is completed
        self.log_dir = "logs"  # Log directory (can be overridden in run_evaluation)

        # Define tool functions
        self.tools = [
            {
                "type": "function",
                "name": "move_up",
                "description": "Move player up. Cannot move if there is a wall or if pushing a box into a wall.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_down",
                "description": "Move player down. Cannot move if there is a wall or if pushing a box into a wall.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_left",
                "description": "Move player left. Cannot move if there is a wall or if pushing a box into a wall.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "move_right",
                "description": "Move player right. Cannot move if there is a wall or if pushing a box into a wall.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]


    def build_full_prompt(self, is_first_call: bool) -> str:
        """
        Build complete prompt including tool descriptions and response format

        Args:
            is_first_call: Whether this is the first call

        Returns:
            Complete prompt text
        """
        if not is_first_call:
            return self.current_prompt

        # Build tool descriptions
        tools_description = ""
        for tool in self.tools:
            if tool.get("type") == "function":
                name = tool.get("name", "")
                description = tool.get("description", "")
                tools_description += f"- {name}: {description}\n"

        if tools_description:
            tools_description = "AVAILABLE ACTIONS:\n\n" + tools_description + "\n"

        # Sokoban game response format
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
- To move up: {"action": "move_up"}
- To move down: {"action": "move_down"}
- To move left: {"action": "move_left"}
- To move right: {"action": "move_right"}

Invalid examples (DO NOT USE):
- {"action": "move_up", "direction": "up"}  # WRONG: contains extra field
- {"action": "move_left", "key": "a"}  # WRONG: contains extra field
- I want to move up: {"action": "move_up"}  # WRONG: contains extra text

IMPORTANT: You must respond with ONLY the JSON object containing EXACTLY the "action" field, no other fields, no additional text.
"""

        return self.current_prompt + "\n\n" + tools_description + response_format

    def capture_game_screenshot(self) -> str:
        """
        Capture game screenshot and encode to base64

        Returns:
            base64 encoded image string
        """
        if not self.input_source:
            raise ValueError("Input source not initialized")

        # Use input source to capture frame data
        frame_data = self.input_source.capture_frame()
        if frame_data is None:
            raise ValueError("Failed to capture frame")

        # Convert numpy array to PIL image
        pil_image = Image.fromarray(frame_data.image)

        # Encode to base64
        return encode_image(pil_image)

    def record_model_response(self, frame_number: int, response, action_taken: str):
        """
        Record model response and decision

        Args:
            frame_number: Frame number
            response: OpenAI API response object
            action_taken: Action taken
        """
        if not self.logger:
            return

        # Import necessary utility functions
        from utils.openai_utils import get_response_text

        # Get response text
        response_text = get_response_text(response)

        # Build response record
        response_record = {
            "frame_number": frame_number,
            "timestamp": time.time(),  # Use system time
            "action_taken": action_taken,
            "response_text": response_text,
            "function_calls": [],
            "raw_response": str(response) if response else ""
        }

        # Extract function call information
        if hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type') and item.type == "function_call":
                    response_record["function_calls"].append({
                        "function_name": item.name,
                        "call_id": getattr(item, 'call_id', 'N/A'),
                        "arguments": getattr(item, 'arguments', {})
                    })

        # Add to response list
        self.model_responses.append(response_record)

        # Log to logger
        self.logger.log_info(f" Model Response - Frame {frame_number}:")
        self.logger.log_info(f"   Action: {action_taken}")
        rsp = str(response_text)
        self.logger.log_info(f"   Response: {rsp}...")
        if response_record['function_calls']:
            self.logger.log_info(f"   Function calls: {[fc['function_name'] for fc in response_record['function_calls']]}")

    async def openai_ai_model(self, frame_data: FrameData) -> Action:
        """
        OpenAI AI model function - Integrated with MLLM framework

        Args:
            frame_data: Frame data

        Returns:
            Action: AI decision action
        """
        self.logger.log_info(f" OpenAI AI Model called for frame {frame_data.frame_number}")

        try:
            # Capture current game screenshot
            screenshot = self.capture_game_screenshot()

            # Build conversation history - use build_conversation_history function
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

            # Call OpenAI API
            response = create_openai_response(
                client=self.client,
                model=self.model,
                conversation_history=conversation_history,
                retry_times=self.retry_times
            )

            # Update conversation history (build_conversation_history already added user input)
            self.conversation_history = conversation_history

            # Add AI response to conversation history
            from utils.openai_utils import get_response_text
            response_text = get_response_text(response)
            self.conversation_history.append({
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": response_text}
                ]
            })

            # Use action_parser module to extract action information
            action_name, action_data = extract_action_info_from_response(response)

            # Process action
            action = None
            if action_name and action_data:
                # Determine key
                key = None

                # If function call method
                if "function_name" in action_data:
                    function_name = action_data["function_name"]
                    action_name = function_name  # Use function name as action name

                    # Get key from key mapping
                    key = SOKOBAN_KEY_MAPPING.get(function_name)

                # If JSON parsing method
                elif "json_data" in action_data:
                    json_data = action_data["json_data"]
                    action_name = json_data.get("action", action_name)

                    # First try to get key from JSON
                    key = json_data.get("key")
                    # If no key, get from action_data
                    if not key:
                        key = action_data.get("key")
                    # If still no key, use key mapping
                    if not key and action_name:
                        key = SOKOBAN_KEY_MAPPING.get(action_name, "w")

                # If key found, create action
                if key:
                    # Extract other information from action_data
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

                # Record AI decision
                if action:
                    self.logger.log_info(f" AI Decision: {action_name} - Frame: {frame_data.frame_number}")
                    # Record model response
                    self.record_model_response(frame_data.frame_number, response, action_name)

            # If no valid action, use default action
            if not action:
                action = Action(
                    type="key_press",
                    key="w",  # Default move up
                    metadata={
                        "frame_number": frame_data.frame_number,
                        "ai_model": "openai_ai",
                        "function_called": "default_move",
                        "response_text": response_text
                    }
                )
                self.logger.log_warning(f" No valid action found, using default action")
                # Record model response for default action
                self.record_model_response(frame_data.frame_number, response, "default_move")

            return action

        except Exception as e:
            self.logger.log_error(f" OpenAI API call failed: {e}", exc_info=True)
            # 
            return Action(
                type="key_press",
                key="w",
                metadata={
                    "frame_number": frame_data.frame_number,
                    "ai_model": "openai_ai",
                    "error": str(e)
                }
            )

    def frame_callback(self, frame_data: FrameData):
        """Frame callback function - record frame data"""
        # Record frame info every 50 frames (reduce log volume)
        if frame_data.frame_number % 50 == 0:
            self.logger.log_info(f" Frame {frame_data.frame_number} captured")

    def state_callback(self, game_state):
        """State callback function - record game state changes"""
        state = game_state.normalized_state

        # Record important state changes
        if state.get("game_won"):
            self.logger.log_info(f" LEVEL COMPLETED! Steps: {state.get('steps_taken', 0)}")
            self.logger.log_info(f"   Distance to goal: {state.get('distance_to_goal', 0)}")
            # Set game completion flag
            self.game_completed = True
            # Stop evaluation engine
            if self.engine:
                self.engine.stop_evaluation()

        elif state.get("game_over") and not state.get("game_won"):
            self.logger.log_info(f" GAME OVER - Failed to complete level")
            self.logger.log_info(f"   Steps taken: {state.get('steps_taken', 0)}")
            self.logger.log_info(f"   Distance to goal: {state.get('distance_to_goal', 0)}")
            # Set game completion flag
            self.game_completed = True
            # Stop evaluation engine
            if self.engine:
                self.engine.stop_evaluation()

        # Record progress every 50 steps (reduce log volume)
        if state.get("steps_taken", 0) % 50 == 0 and state.get("steps_taken", 0) > 0:
            self.logger.log_info(f" Progress: Step {state.get('steps_taken', 0)}, "
                               f"Distance: {state.get('distance_to_goal', 0)}")

    def action_callback(self, action: Action):
        """Action callback function - record AI decisions"""
        frame_number = action.metadata.get("frame_number", 0)
        function_called = action.metadata.get("function_called", "unknown")

        # Record basic info every 20 actions (reduce log volume)
        if frame_number % 20 == 0:
            self.logger.log_info(f" AI Action - Frame {frame_number}: {function_called}")

    def save_model_responses(self):
        """Save all model responses to JSON file"""
        if not self.model_responses:
            self.logger.log_info(" No model responses to save")
            return

        try:
            # Ensure log directory exists
            os.makedirs(self.log_dir, exist_ok=True)

            # Create response record file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_dir}/sokoban_opengl_model_responses_level{self.current_level}_{timestamp}.json"

            # Build complete response data
            response_data = {
                "evaluation_info": {
                    "model": self.model,
                    "level": self.current_level,
                    "total_responses": len(self.model_responses),
                    "timestamp": timestamp,
                    "successful_actions": sum(1 for r in self.model_responses if r["action_taken"] != "default_move"),
                    "default_actions": sum(1 for r in self.model_responses if r["action_taken"] == "default_move"),
                    "renderer": "opengl"
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

    async def run_evaluation(self, level: int = 0, max_steps: int = 200, initial_prompt: str = None, log_dir: str = "logs") -> Dict[str, Any]:
        """
        Run evaluation - using MLLM framework

        Args:
            level: Game level (0-4)
            max_steps: Maximum step limit
            initial_prompt: Initial prompt

        Returns:
            Evaluation results
        """
        # Reset game completion flag
        self.game_completed = False

        # Save log directory and level for use by other methods
        self.log_dir = log_dir
        self.current_level = level

        # Setup logging system
        self.logger = setup_global_logger(
            name="sokoban_opengl_openai_evaluation",
            log_level=logging.INFO,
            log_to_file=True,
            log_dir=log_dir
        )

        self.logger.log_info(" Starting Sokoban Game OpenGL OpenAI API Evaluation")
        self.logger.log_info(f"   AI Model: OpenAI {self.model}")
        self.logger.log_info(f"   Level: {level} - {LEVEL_CONFIG['level_descriptions'][level]}")
        self.logger.log_info(f"   Max steps: {max_steps}")
        self.logger.log_info(f"   Headless mode: True")
        self.logger.log_info(f"   Video recording: Enabled")
        self.logger.log_info(f"   Renderer: OpenGL")

        # Set up prompt
        if initial_prompt is None:
            self.current_prompt = SYSTEM_PROMPT
        else:
            self.current_prompt = initial_prompt

        self.logger.log_info(f"   Prompt type: {'custom' if initial_prompt else 'default'}")

        # Create Sokoban game instance
        self.game = SokobanGame(level=level)

        # Create Pygame OpenGL input source
        self.input_source = PygameOpenGLInputSource()

        # Initialize input source (headless mode with OpenGL support)
        success = self.input_source.initialize(
            game_module=self.game,
            headless=True,
            screen_size=(800, 600)
        )

        if not success:
            self.logger.log_error(" Failed to initialize OpenGL input source for Sokoban")
            return None

        self.logger.log_input_source_init(self.input_source.get_info())

        # Create evaluation engine (enable video recording)
        self.engine = EvaluationEngine(
            input_source=self.input_source,
            decision_frequency=2.0,  # 2 seconds per decision, giving OpenAI API enough time
            record_video=True,       # Enable video recording
            video_fps=0.5,           # One frame per decision (1 frame / 2 seconds)
            video_output_dir="videos"
        )

        # Add callback functions
        self.engine.add_frame_callback(self.frame_callback)
        self.engine.add_state_callback(self.state_callback)
        self.engine.add_action_callback(self.action_callback)

        self.logger.log_evaluation_start(max_steps=max_steps, max_duration=None)

        # Run evaluation
        try:
            stats = await self.engine.run_evaluation(
                ai_model_func=self.openai_ai_model,
                max_steps=max_steps,
                max_duration=None,
                session_name=f"sokoban_opengl_level_{level}"
            )

            # Record final statistics
            self.logger.log_evaluation_end(stats)
            self.logger.log_performance_metrics(stats)

            # Get final game state
            final_state = self.input_source.get_game_state()
            if final_state:
                state = final_state.normalized_state
                self.logger.log_info("\n=== FINAL GAME RESULTS ===")
                self.logger.log_info(f"Success: {state.get('success', False)}")
                self.logger.log_info(f"Steps taken: {state.get('steps_taken', 0)}")
                self.logger.log_info(f"Distance to goal: {state.get('distance_to_goal', 0)}")
                self.logger.log_info(f"Level: {state.get('level', 0)}")

            # Video recording information
            if os.path.exists("videos"):
                video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
                if video_files:
                    self.logger.log_info(f" Video files recorded: {len(video_files)}")

            return {
                "success": state.get('success', False) if final_state else False,
                "steps_taken": state.get('steps_taken', 0) if final_state else 0,
                "max_steps": max_steps,
                "level": level,
                "distance_to_goal": state.get('distance_to_goal', 0) if final_state else 0,
                "stats": stats
            }

        except Exception as e:
            self.logger.log_error(f" Sokoban evaluation failed: {e}", exc_info=True)
            return None
        finally:
            # Save model response records
            self.save_model_responses()

            # Clean up resources
            self.input_source.close()
            self.logger.log_info(" Sokoban OpenGL OpenAI evaluation completed")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sokoban Game OpenGL OpenAI API Evaluation')
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help='Game level (0-4, default: 0)')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum step limit (default: 200)')
    parser.add_argument('--custom-prompt', type=str,
                       help='Custom prompt (overrides default prompt)')
    parser.add_argument('--api-base-url', type=str, default=OPENAI_CONFIG['api_base_url'],
                       help='OpenAI API base URL')
    parser.add_argument('--api-key', type=str, default=OPENAI_CONFIG['api_key'],
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default=OPENAI_CONFIG['model'],
                       help='Model name to use')
    parser.add_argument('--log-dir', type=str, default=OUTPUT_CONFIG.get('log_dir', 'logs'),
                       help=f'Log output directory (default: {OUTPUT_CONFIG.get("log_dir", "logs")})')
    parser.add_argument('--retry-times', type=int, default=EVALUATION_CONFIG.get('retry_times', 3),
                       help=f'Max API retry times on failure (default: {EVALUATION_CONFIG.get("retry_times", 3)})')

    return parser.parse_args()


async def main():
    """Main function - demonstrates how to use the evaluator"""
    args = parse_arguments()

    print("=" * 60)
    print("Sokoban Game OpenGL OpenAI API Evaluation (MLLM Framework Version)")
    print("=" * 60)
    print(f"Configuration Parameters:")
    print(f"  Game Level: {args.level} - {LEVEL_CONFIG['level_descriptions'][args.level]}")
    print(f"  Maximum Steps: {args.max_steps}")
    print(f"  API Model: {args.model}")
    print(f"  Renderer: OpenGL")
    print("=" * 60)

    # Create evaluator
    evaluator = SokobanOpenGLOpenAIEvaluator(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model=args.model
    )
    evaluator.retry_times = args.retry_times

    # Select prompt
    if args.custom_prompt:
        initial_prompt = args.custom_prompt
        print("Using custom prompt")
    else:
        initial_prompt = SYSTEM_PROMPT
        print("Using default prompt")

    # Run evaluation
    result = await evaluator.run_evaluation(
        level=args.level,
        max_steps=args.max_steps,
        initial_prompt=initial_prompt,
        log_dir=args.log_dir
    )

    # Output detailed results
    if result:
        print("\n" + "="*60)
        print("Detailed Evaluation Results:")
        print(f"Level Completed: {'Yes' if result['success'] else 'No'}")
        print(f"Steps Used: {result['steps_taken']}/{result['max_steps']}")
        print(f"Final Distance to Goal: {result['distance_to_goal']} cells")
        print(f"Level: {result['level']}")

        # Display performance statistics
        stats = result['stats']
        print(f"\n Performance Statistics:")
        print(f"  Total Frames: {stats['total_frames']}")
        print(f"  Total Time: {stats['total_time']:.2f}s")
        print(f"  Average FPS: {stats['actual_fps']:.2f}")
        print(f"  Average Inference Time: {stats['avg_inference_time']:.3f}s")

        # Check log and video files
        logger = get_logger()
        log_file = logger.get_log_file_path()
        if log_file:
            print(f"\n Log File: {log_file}")

        if os.path.exists("videos"):
            video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')]
            if video_files:
                print(f" Video Files:")
                for video_file in video_files:
                    print(f"   - videos/{video_file}")

        print("="*60)

        # Save result summary
        summary = {
            "success": result["success"],
            "steps_taken": result["steps_taken"],
            "max_steps": result["max_steps"],
            "level": result["level"],
            "distance_to_goal": result["distance_to_goal"],
            "model": args.model,
            "renderer": "opengl",
            "total_time": stats['total_time'],
            "total_frames": stats['total_frames']
        }

        # Ensure log directory exists
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"{args.log_dir}/sokoban_opengl_openai_summary_level{args.level}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Result summary saved to: {args.log_dir}/sokoban_opengl_openai_summary_level{args.level}_{timestamp}.json")
    else:
        print("\n Evaluation failed, please check log file for details")


if __name__ == "__main__":
    asyncio.run(main())
