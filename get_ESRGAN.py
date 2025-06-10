import gdown
import os

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


def upscale_video():
  