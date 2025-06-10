import os
from pathlib import Path
import subprocess
import uuid
import cv2

from .adjust_properties import adjust_image_properties

BASE_DIR = Path(__file__).parents[2]
UPLOAD_DIR = "uploads"

async def deoldify_video(input_video_path: str, output_video_path: str, brightness, sharpness, contrast) -> str:
    # """
    # Converts a video to black and white while preserving the original audio.
    # """
    # if not os.path.exists(input_video_path):
        # raise FileNotFoundError(f"Input video file '{input_video_path}' not found.")
#
    # try:
        # Ensure the output path is unique
        # if os.path.exists(output_video_path):
            # output_video_path = f"{os.path.splitext(output_video_path)[0]}_{uuid.uuid4().hex}.mp4"
#
        # Extract audio
        # temp_audio_path = f"{output_video_path}_audio.aac"
        # has_audio = False
        # extract_audio_command = [
            # "ffmpeg", "-y", "-i", input_video_path, "-q:a", "0", "-map", "a?", temp_audio_path
        # ]
        # try:
            # subprocess.run(extract_audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # if os.path.exists(temp_audio_path):
                # print(f"Audio extracted to {temp_audio_path}")
                # has_audio = True
        # except:
            # print("No audio stream found. Proceeding without audio.")
#
        # Process video to grayscale
        # cap = cv2.VideoCapture(input_video_path)
        # if not cap.isOpened():
            # raise RuntimeError("Failed to open video file for processing.")
#
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cap.get(cv2.CAP_PROP_FPS)
#
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
#
        # def process_frame(frame, brightness, sharpness, contrast):
            # bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # return adjust_image_properties(bw_frame, brightness=int(brightness), sharpness=int(sharpness), contrast=int(contrast))
#
        # while True:
            # ret, frame = cap.read()
            # if not ret:
                # break
            # gray_frame = process_frame(frame, brightness, sharpness, contrast)
            # out.write(gray_frame)
#
        # cap.release()
        # out.release()
#
        # final_out_video_path = f"{os.path.splitext(output_video_path)[0]}_bw_{uuid.uuid4().hex[:6]}.mp4"
#
        # if has_audio:
            # merge_command = [
                # "ffmpeg", "-y",
                # "-i", output_video_path,
                # "-i", temp_audio_path,
                # "-map", "0:v?",
                # "-map", "1:a?",
                # "-vcodec", "libx264",
                # "-c:a", "aac",
                # "-shortest",
                # final_out_video_path
            # ]
            # subprocess.run(merge_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # else:
            # finalise_command = [
                # "ffmpeg", "-y",
                # "-i", output_video_path,
                # "-c:v", "libx264",
                # "-b:a", "192k",
                # "-preset", "fast",
                # final_out_video_path
            # ]
            # subprocess.run(finalise_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#
        # if has_audio:
            # os.remove(temp_audio_path)
        # os.remove(output_video_path)
#
        # return final_out_video_path
#
#
    # except subprocess.CalledProcessError as e:
        # raise RuntimeError(f"FFmpeg command failed: {e.stderr.decode()}")
#
    # except Exception as e:
        # raise RuntimeError(f"An error occurred: {str(e)}")


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
