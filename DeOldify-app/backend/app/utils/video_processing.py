import os
from pathlib import Path
import subprocess
import uuid
from Inference_pipeline.main_infernece import restore_video_pipeline
import cv2

from .adjust_properties import adjust_image_properties

BASE_DIR = Path(__file__).parents[2]
UPLOAD_DIR = "uploads"

async def deoldify_video(input_video_path: str, output_video_path: str, brightness: float, sharpness: float, contrast: float) -> str:
    """
    Processes a video using DeOldify + RIFE + ESRGAN and returns the output path.

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Desired path to save the final video.
        brightness (float): Brightness adjustment factor.
        sharpness (float): Sharpness adjustment factor.
        contrast (float): Contrast adjustment factor.

    Returns:
        str: Path to the final restored video.
    """

    # Run the restoration pipeline
    final_path = restore_video_pipeline(input_video_path)

    # Optionally move or rename the output to match output_video_path
    import shutil
    shutil.move(final_path, output_video_path)

    return output_video_path


async def compress_video(input_video_path: str, output_video_path: str, resolution="1280x720", bitrate="1M") -> str:
    """
    Compresses a video to reduce its size.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file '{input_video_path}' not found.")

    if os.path.exists(output_video_path):
        output_video_path = f"{os.path.splitext(output_video_path)[0]}_compressed_{uuid.uuid4().hex}.mp4"

    compress_command = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vf", f"scale={resolution}",
        "-b:v", bitrate,
        "-c:a", "aac",
        "-strict", "experimental",
        output_video_path
    ]

    try:
        subprocess.run(compress_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_video_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video compression failed: {e.stderr.decode()}")


def check_resolution(video_path: str, threshold_width=1280, threshold_height=720) -> bool:
    """
    Checks if the video's resolution exceeds the specified threshold.
    """
    import json
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    metadata = json.loads(result.stdout)
    width = metadata["streams"][0]["width"]
    height = metadata["streams"][0]["height"]
    return width < threshold_width or height < threshold_height
