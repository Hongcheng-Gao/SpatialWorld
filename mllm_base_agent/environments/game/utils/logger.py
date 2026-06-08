"""
    
"""
import logging
import os
from datetime import datetime
from typing import Optional


class GameEvaluationLogger:
    """        """

    def __init__(self, name: str = "game_evaluation",
                 log_level: int = logging.INFO,
                 log_to_file: bool = True,
                 log_dir: str = "logs"):
        """
               

        Args:
            name:      
            log_level:     
            log_to_file:        
            log_dir:     
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        #         
        self.logger.handlers.clear()

        #       
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        #       
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        #      
        if log_to_file:
            self._setup_file_handler(log_dir, formatter, log_level)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _setup_file_handler(self, log_dir: str, formatter: logging.Formatter,
                           log_level: int):
        """       """
        try:
            #       
            os.makedirs(log_dir, exist_ok=True)

            #         
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"session_{timestamp}.log")

            #      
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file
        except Exception as e:
            print(f"Failed to setup file logging: {e}")

    def log_framework_init(self, framework_info: dict):
        """         """
        self.logger.info("Framework initialized")
        self.logger.debug(f"Framework info: {framework_info}")

    def log_input_source_init(self, source_info: dict):
        """          """
        self.logger.info(f"Input source initialized: {source_info['type']}")
        self.logger.debug(f"Input source capabilities: {source_info['capabilities']}")

    def log_frame_capture(self, frame_data):
        """       """
        self.logger.debug(
            f"Frame captured - Frame: {frame_data.frame_number}, "
            f"Timestamp: {frame_data.timestamp:.3f}, "
            f"Size: {frame_data.image.shape}"
        )

    def log_ai_decision(self, frame_data, action, inference_time: float):
        """  AI    """
        self.logger.info(
            f"AI Decision - Frame: {frame_data.frame_number}, "
            f"Action: {action.type}, "
            f"Inference time: {inference_time:.3f}s"
        )
        if action.metadata:
            self.logger.debug(f"Action metadata: {action.metadata}")

    def log_action_execution(self, action, success: bool, execution_time: float):
        """        """
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"Action executed - Type: {action.type}, "
            f"Success: {success}, "
            f"Execution time: {execution_time:.3f}s"
        )

    def log_game_state(self, game_state):
        """        """
        if game_state.normalized_state:
            self.logger.debug(
                f"Game state updated - Normalized: {game_state.normalized_state}"
            )

    def log_performance_metrics(self, metrics: dict):
        """      """
        self.logger.info(
            f"Performance - FPS: {metrics.get('actual_fps', 0):.2f}, "
            f"Capture: {metrics.get('avg_capture_time', 0):.3f}s, "
            f"Inference: {metrics.get('avg_inference_time', 0):.3f}s, "
            f"Action: {metrics.get('avg_action_time', 0):.3f}s"
        )

    def log_evaluation_start(self, max_steps: Optional[int], max_duration: Optional[float]):
        """      """
        self.logger.info("Evaluation started")
        conditions = []
        if max_steps:
            conditions.append(f"max_steps={max_steps}")
        if max_duration:
            conditions.append(f"max_duration={max_duration}s")
        if conditions:
            self.logger.info(f"Stop conditions: {'; '.join(conditions)}")

    def log_evaluation_end(self, stats: dict):
        """      """
        self.logger.info("Evaluation completed")
        self.logger.info(
            f"Final stats - Total frames: {stats.get('total_frames', 0)}, "
            f"Total time: {stats.get('total_time', 0):.2f}s, "
            f"Average FPS: {stats.get('actual_fps', 0):.2f}"
        )

    def log_error(self, error_msg: str, exc_info: bool = False):
        """      """
        self.logger.error(error_msg, exc_info=exc_info)

    def log_warning(self, warning_msg: str):
        """      """
        self.logger.warning(warning_msg)

    def log_debug(self, debug_msg: str):
        """      """
        self.logger.debug(debug_msg)

    def log_info(self, info_msg: str):
        """      """
        self.logger.info(info_msg)

    def get_log_file_path(self) -> Optional[str]:
        """        """
        return getattr(self, 'log_file', None)


#        
_global_logger = None


def setup_global_logger(**kwargs):
    """       """
    global _global_logger
    _global_logger = GameEvaluationLogger(**kwargs)
    return _global_logger


def get_logger() -> GameEvaluationLogger:
    """       """
    global _global_logger
    if _global_logger is None:
        _global_logger = GameEvaluationLogger()
    return _global_logger