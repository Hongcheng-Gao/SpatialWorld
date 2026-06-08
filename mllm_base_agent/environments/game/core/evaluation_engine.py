"""
Main Evaluation Loop Engine
"""
import time
import asyncio
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from .input_source import GameInputSource
from .data_classes import FrameData, GameState, Action


# Import video recorder (delayed import to avoid dependency issues)
def _import_video_recorder():
    try:
        import sys
        import os
        # Add project root directory to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from utils.video_recorder import VideoRecorder
        return VideoRecorder
    except ImportError as e:
        print(f"Failed to import VideoRecorder: {e}")
        return None


class EvaluationEngine:
    """Evaluation Engine"""

    def __init__(self, input_source: GameInputSource,
                 decision_frequency: float = 10.0,
                 record_video: bool = False,
                 video_fps: int = 10,
                 video_output_dir: str = "videos"):
        """
        Initialize evaluation engine

        Args:
            input_source: Game input source object
            decision_frequency: Decision frequency (Hz)
            record_video: Whether to record video
            video_fps: Video frame rate
            video_output_dir: Video output directory
        """
        self.input_source = input_source
        self.decision_frequency = decision_frequency
        self.frame_interval = 1.0 / decision_frequency

        #     
        self.frame_callbacks = []
        self.state_callbacks = []
        self.action_callbacks = []

        #     
        self.record_video = record_video
        self.video_recorder = None
        if record_video:
            VideoRecorder = _import_video_recorder()
            if VideoRecorder:
                self.video_recorder = VideoRecorder(
                    output_dir=video_output_dir,
                    fps=video_fps
                )
            else:
                print("Warning: Video recording not available. Install opencv-python to enable video recording.")
                self.record_video = False

        #     
        self.stats = {
            "total_frames": 0,
            "total_time": 0.0,
            "capture_times": [],
            "inference_times": [],
            "action_times": [],
            "frame_times": []
        }

        #     
        self.is_running = False
        self.start_time = None

    def add_frame_callback(self, callback: Callable[[FrameData], None]):
        """
               

        Args:
            callback:     ，  FrameData  
        """
        self.frame_callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[GameState], None]):
        """
                

        Args:
            callback:     ，  GameState  
        """
        self.state_callbacks.append(callback)

    def add_action_callback(self, callback: Callable[[Action], None]):
        """
                

        Args:
            callback:     ，  Action  
        """
        self.action_callbacks.append(callback)

    async def run_evaluation(self, ai_model_func: Callable[[FrameData], Action],
                           max_steps: Optional[int] = None,
                           max_duration: Optional[float] = None,
                           session_name: Optional[str] = None) -> Dict[str, Any]:
        """
               

        Args:
            ai_model_func: AI    ，  FrameData，  Action
            max_steps:     
            max_duration:       （ ）
            session_name:     ，      

        Returns:
            Dict[str, Any]:       
        """
        self.is_running = True
        self.start_time = time.time()
        step_count = 0
        first_frame = None

        try:
            while self.is_running:
                loop_start_time = time.time()

                #       
                if max_steps and step_count >= max_steps:
                    break
                if max_duration and (time.time() - self.start_time) >= max_duration:
                    break

                #    
                capture_start = time.time()
                frame_data = self.input_source.capture_frame()
                capture_time = time.time() - capture_start

                if frame_data is None:
                    print("Failed to capture frame, skipping step")
                    continue

                #               
                if first_frame is None:
                    first_frame = frame_data

                #       （         ）
                if (self.record_video and self.video_recorder and
                    not self.video_recorder.is_active() and first_frame):
                    try:
                        self.video_recorder.start_recording(first_frame, session_name)
                    except Exception as e:
                        print(f"Failed to start video recording: {e}")
                        self.record_video = False

                #      
                for callback in self.frame_callbacks:
                    try:
                        callback(frame_data)
                    except Exception as e:
                        print(f"Frame callback error: {e}")

                # AI   -         
                inference_start = time.time()
                try:
                    action = await ai_model_func(frame_data)
                    inference_time = time.time() - inference_start
                except Exception as e:
                    print(f"AI model error: {e}")
                    action = None
                    inference_time = 0.0

                #       -          ，       
                if (self.record_video and self.video_recorder and
                    self.video_recorder.is_active() and action is not None):
                    self.video_recorder.add_frame(frame_data)

                #     
                action_time = 0.0
                if action is not None:
                    action_start = time.time()
                    success = self.input_source.execute_action(action)
                    action_time = time.time() - action_start

                    #       
                    if success:
                        for callback in self.action_callbacks:
                            try:
                                callback(action)
                            except Exception as e:
                                print(f"Action callback error: {e}")

                #       
                game_state = self.input_source.get_game_state()
                if game_state:
                    for callback in self.state_callbacks:
                        try:
                            callback(game_state)
                        except Exception as e:
                            print(f"State callback error: {e}")

                #     
                self._update_stats(capture_time, inference_time, action_time,
                                 time.time() - loop_start_time)
                step_count += 1

                #          ，          
                #         ，           
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Evaluation interrupted by user")
        except Exception as e:
            print(f"Evaluation error: {e}")
        finally:
            #       
            if self.record_video and self.video_recorder:
                self.video_recorder.stop_recording()
                video_file = self.video_recorder.get_video_file()
                if video_file:
                    print(f"Video saved to: {video_file}")

            self.is_running = False
            self.stats["total_time"] = time.time() - self.start_time
            self.stats["total_frames"] = step_count

        return self._get_final_stats()

    def stop_evaluation(self):
        """    """
        self.is_running = False

    def _update_stats(self, capture_time: float, inference_time: float,
                     action_time: float, frame_time: float):
        """      """
        self.stats["capture_times"].append(capture_time)
        self.stats["inference_times"].append(inference_time)
        self.stats["action_times"].append(action_time)
        self.stats["frame_times"].append(frame_time)

    def _get_final_stats(self) -> Dict[str, Any]:
        """        """
        stats = self.stats.copy()

        #       
        if stats["capture_times"]:
            stats["avg_capture_time"] = sum(stats["capture_times"]) / len(stats["capture_times"])
            stats["max_capture_time"] = max(stats["capture_times"])
        else:
            stats["avg_capture_time"] = 0
            stats["max_capture_time"] = 0

        if stats["inference_times"]:
            stats["avg_inference_time"] = sum(stats["inference_times"]) / len(stats["inference_times"])
            stats["max_inference_time"] = max(stats["inference_times"])
        else:
            stats["avg_inference_time"] = 0
            stats["max_inference_time"] = 0

        if stats["action_times"]:
            stats["avg_action_time"] = sum(stats["action_times"]) / len(stats["action_times"])
            stats["max_action_time"] = max(stats["action_times"])
        else:
            stats["avg_action_time"] = 0
            stats["max_action_time"] = 0

        if stats["frame_times"]:
            stats["avg_frame_time"] = sum(stats["frame_times"]) / len(stats["frame_times"])
            stats["max_frame_time"] = max(stats["frame_times"])
        else:
            stats["avg_frame_time"] = 0
            stats["max_frame_time"] = 0

        #       
        if stats["total_time"] > 0:
            stats["actual_fps"] = stats["total_frames"] / stats["total_time"]
        else:
            stats["actual_fps"] = 0

        #       
        del stats["capture_times"]
        del stats["inference_times"]
        del stats["action_times"]
        del stats["frame_times"]

        return stats

    def get_current_stats(self) -> Dict[str, Any]:
        """      """
        if not self.is_running:
            return self._get_final_stats()

        current_stats = self.stats.copy()
        current_time = time.time() - self.start_time

        if current_stats["capture_times"]:
            current_stats["current_capture_time"] = current_stats["capture_times"][-1]
        else:
            current_stats["current_capture_time"] = 0

        if current_stats["inference_times"]:
            current_stats["current_inference_time"] = current_stats["inference_times"][-1]
        else:
            current_stats["current_inference_time"] = 0

        current_stats["elapsed_time"] = current_time
        current_stats["current_fps"] = len(current_stats["frame_times"]) / current_time if current_time > 0 else 0

        return current_stats