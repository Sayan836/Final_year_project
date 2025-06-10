import gdown
import os
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from ESRGAN import RealESRGANer
import numpy as np

def download_esrgan_model():
    """
    Downloads the ESRGAN model from Google Drive into /final_year_project/models
    if it's not already present.
    """
    # Google Drive file ID and destination
    file_id = '1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_folder = './models'
    output_file = os.path.join(output_folder, 'RRDB_ESRGAN_x4.pth')

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Download only if file doesn't exist
    if not os.path.exists(output_file):
        print("Downloading ESRGAN model...")
        gdown.download(url, output_file, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists at:", output_file)

    return output_file


def upscale_video(input_path, output_path, model_path, scale=4, tile=0, gpu_id=0):
    """
    Upscale a video using a trained Real-ESRGAN model.

    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the upscaled video
        model_path (str): Path to the trained model
        scale (int): Upscaling factor (default: 4)
        tile (int): Tile size for processing large images (default: 0)
        gpu_id (int): GPU device ID to use (default: 0)
    """
    # Create model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=True,  # Use half precision for faster processing
        gpu_id=gpu_id
    )

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * scale, height * scale)
    )

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale frame
            output, _ = upsampler.enhance(frame, outscale=scale)

            # Write frame to output video
            out.write(output)

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")

        print(f"Successfully upscaled video. Saved to: {output_path}")

    except RuntimeError as error:
        print('Error:', error)
        print('If you encounter CUDA out of memory, try to set tile with a smaller number.')

    finally:
        # Release resources
        cap.release()
        out.release()
    return output_path