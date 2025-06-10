#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
import torch
if not torch.cuda.is_available():
    print('GPU not available.')
from os import path
import os
import fastai
from deoldify.visualize import *
from pathlib import Path
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
from dotenv import load_dotenv

from get_RIFE import *
from get_ESRGAN import *

load_dotenv()
download_rife_model()
esrgan_model=download_esrgan_model()


def restore_video_pipeline(input_video_path, rife_model_dir='train_log', exp=2, scale=1.0, fp16=True):
    """
    Applies a full restoration pipeline to a video: colorization (DeOldify), interpolation (RIFE), and upscaling (ESRGAN).

    Args:
        input_video_path (str): Path to the input video.
        rife_model_dir (str): Directory of the RIFE model checkpoint. Default is 'train_log'.
        exp (int): The exponent for frame interpolation. exp=2 means 4x interpolation.
        scale (float): Scale factor used during interpolation. Default is 1.0.
        fp16 (bool): Whether to use fp16 precision during RIFE inference.

    Returns:
        str: Path to the final processed video.
    """
    # Step 1: Colorize using DeOldify
    print("[1/3] Colorizing video...")
    colorizer = get_video_colorizer()
    output_path = Path(input_video_path).stem + "_colorized.mp4"
    colorized_path = colorizer.colorize_from_file_name(input_video_path,output_path=output_path, render_factor=35, results_dir='colorized_videos')
    print(f"Colorized video saved at: {colorized_path}")

    # Step 2: Interpolate using RIFE
    print("[2/3] Interpolating video frames...")
    interpolated_path = interpolate_video(
        video_path=colorized_path,
        output_path=Path(colorized_path).stem + "_interpolated.mp4",
        model_dir=rife_model_dir,
        exp=exp,
        scale=scale,
        fp16=fp16
    )
    print(f"Interpolated video saved at: {interpolated_path}")

    # Step 3: Upscale using ESRGAN
    print("[3/3] Upscaling video resolution...")
    output_path= Path(interpolated_path).stem + "_upscaled.mp4"
    final_output_path = upscale_video(interpolated_path, output_path, esrgan_model, scale=4, tile=0, gpu_id=0)
    print(f"Upscaled video saved at: {final_output_path}")

    return final_output_path

if __name__ == "__main__":
    final_video = restore_video_pipeline("/path/to/input_video.mp4")
    print("Final processed video at:", final_video)
