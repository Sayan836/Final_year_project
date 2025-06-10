import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

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

def upscale_image(input_path, output_path, model_path, scale=4, tile=0, gpu_id=0):
    """
    Upscale an image using a trained Real-ESRGAN model.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the upscaled image
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

    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    try:
        # Upscale image
        output, _ = upsampler.enhance(img, outscale=scale)

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)
        print(f"Successfully upscaled image. Saved to: {output_path}")

    except RuntimeError as error:
        print('Error:', error)
        print('If you encounter CUDA out of memory, try to set tile with a smaller number.')

if __name__ == '__main__':
    # Example usage
    model_path = r"E:\Real-ESRGAN\experiments\train_test_dataset_4x\models\latest_G.pth"

    # For image upscaling
    input_image = "input.jpg"  # Replace with your input image path
    output_image = "output_upscaled.png"  # Replace with your desired output path

    # For video upscaling
    input_video = "input.mp4"  # Replace with your input video path
    output_video = "output_upscaled.mp4"  # Replace with your desired output path

    # Choose whether to process image or video
    process_video = True  # Set to False for image processing

    if process_video:
        upscale_video(
            input_path=input_video,
            output_path=output_video,
            model_path=model_path,
            scale=4,  # 4x upscaling
            tile=0,   # No tiling (adjust if you get out of memory errors)
            gpu_id=0  # Use first GPU
        )
    else:
        upscale_image(
            input_path=input_image,
            output_path=output_image,
            model_path=model_path,
            scale=4,  # 4x upscaling
            tile=0,   # No tiling (adjust if you get out of memory errors)
            gpu_id=0  # Use first GPU
        )