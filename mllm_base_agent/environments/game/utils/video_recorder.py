"""
Video Recorder
"""
import cv2
import os
from datetime import datetime
from typing import Optional

#            
def _import_frame_data():
    try:
        from core.data_classes import FrameData
        return FrameData
    except ImportError:
        return None


class VideoRecorder:
    """Video recorder for saving evaluation process videos"""

    def __init__(self, output_dir: str = "videos", fps: int = 10):
        """
        Initialize video recorder

        Args:
            output_dir: Video output directory
            fps: Video frame rate
        """
        self.output_dir = output_dir
        self.fps = fps
        self.video_writer = None
        self.is_recording = False
        self.video_file = None

        #       
        os.makedirs(output_dir, exist_ok=True)

    def start_recording(self, frame_data, session_name: Optional[str] = None):
        """
        Start recording video

        Args:
            frame_data: First frame data for video parameters
            session_name: Session name for filename generation
        """
        if self.is_recording:
            return

        # Generate filename
        if session_name:
            filename = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        self.video_file = os.path.join(self.output_dir, filename)

        # Get video parameters
        height, width = frame_data.image.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_file,
            fourcc,
            self.fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_file}")

        self.is_recording = True
        print(f"Started recording video: {self.video_file}")

    def add_frame(self, frame_data):
        """
        Add frame to video

        Args:
            frame_data: Frame data
        """
        if not self.is_recording or self.video_writer is None:
            return

        try:
            # Ensure correct image format (BGR for OpenCV)
            frame = frame_data.image.copy()

            # If image is RGB, convert to BGR
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.video_writer.write(frame)
        except Exception as e:
            print(f"Failed to write frame to video: {e}")

    def stop_recording(self):
        """Stop recording video"""
        if not self.is_recording:
            return

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.is_recording = False
        print(f"Stopped recording video: {self.video_file}")

    def is_active(self) -> bool:
        """Check if currently recording"""
        return self.is_recording

    def get_video_file(self) -> Optional[str]:
        """Get recorded video file path"""
        return self.video_file