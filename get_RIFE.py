import gdown
import os
import py7zr

def download_rife_model():
    """
    Downloads and extracts the RIFE trained model (v3.6) from Google Drive
    into /Final_year_project/RIFE/train_log directory.
    """
    # Target directory
    output_dir = './Final_year_project/RIFE/train_log'
    os.makedirs(output_dir, exist_ok=True)

    # Google Drive file ID and output ZIP path
    file_id = '1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_'
    zip_path = os.path.join(output_dir, 'RIFE_trained_model_v3.6.7z')

    # Download the .7z file using gdown
    if not os.path.exists(zip_path):
        print("Downloading RIFE model...")
        gdown.download(id=file_id, output=zip_path, quiet=False)
        print("Download complete.")
    else:
        print("7z archive already exists:", zip_path)

    # Extract .7z file using py7zr
    print("Extracting the RIFE model...")
    with py7zr.SevenZipFile(zip_path, mode='r') as archive:
        archive.extractall(path=output_dir)
    print("Extraction complete.")

    return output_dir

def interpolate_video(
    video_path,
    output_path=None,
    model_dir="RIFE/train_log",
    scale=1.0,
    exp=1,
    fps_override=None,
    fp16=False,
    montage=False,
    UHD=False,
    img_input=None,
    png_output=False,
    ext="mp4"
):
    import os
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch.nn import functional as F
    from RIFE.model.pytorch_msssim import ssim_matlab
    from queue import Queue
    import _thread
    import skvideo.io
    import warnings

    warnings.filterwarnings("ignore")

    # ========== DEVICE SETUP ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # ========== LOAD MODEL ==========
    try:
        from RIFE.model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded v2.x HD model.")
    except:
        try:
            from RIFE.train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v3.x HD model.")
        except:
            from RIFE.model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model.")

    model.eval()
    model.device()

    # ========== VIDEO READER ==========
    assert video_path is not None or img_input is not None, "Provide a video or image folder"

    if UHD and scale == 1.0:
        scale = 0.5
    assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]

    if video_path:
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if fps_override is None:
            fps_out = fps * (2 ** exp)
        else:
            fps_out = fps_override
        videogen = skvideo.io.vreader(video_path)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        base_path, _ = os.path.splitext(video_path)
        if output_path is None:
            output_path = f"{base_path}_{2 ** exp}X_{int(np.round(fps_out))}.{ext}"
    else:
        # Image sequence
        videogen = sorted([f for f in os.listdir(img_input) if f.endswith(".png")], key=lambda x: int(x[:-4]))
        tot_frame = len(videogen)
        lastframe = cv2.imread(os.path.join(img_input, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
        png_output = True

    h, w, _ = lastframe.shape
    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad_image(img):
        return F.pad(img, padding).half() if fp16 else F.pad(img, padding)

    def make_inference(I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        return [*first_half, middle, *second_half] if n % 2 else [*first_half, *second_half]

    read_buffer = Queue(maxsize=500)
    write_buffer = Queue(maxsize=500)

    def build_read_buffer():
        try:
            for frame in videogen:
                if img_input:
                    frame = cv2.imread(os.path.join(img_input, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def clear_write_buffer():
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if png_output:
                if not os.path.exists('vid_out'):
                    os.mkdir('vid_out')
                cv2.imwrite(f'vid_out/{cnt:07d}.png', item[:, :, ::-1])
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])

    # Video writer
    if not png_output:
        vid_out = cv2.VideoWriter(output_path, fourcc, fps_out, (w, h))
    else:
        vid_out = None

    _thread.start_new_thread(build_read_buffer, ())
    _thread.start_new_thread(clear_write_buffer, ())

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    pbar = tqdm(total=tot_frame)
    temp = None
    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)

        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.996:
            frame = read_buffer.get()
            if frame is None:
                break
            temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        output = make_inference(I0, I1, 2 ** exp - 1) if ssim >= 0.2 else [I0] * ((2 ** exp) - 1)

        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame

    write_buffer.put(lastframe)
    write_buffer.put(None)
    import time
    while not write_buffer.empty():
        time.sleep(0.1)
    pbar.close()

    if vid_out:
        vid_out.release()

    return output_path
