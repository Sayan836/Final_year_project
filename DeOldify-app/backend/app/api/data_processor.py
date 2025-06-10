import hashlib
from io import BytesIO
import os
from pathlib import Path
import time
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..utils.image_processing import process_image
from ..utils.video_processing import deoldify_video, compress_video, check_resolution
from ..functions.check_image_or_video import check_file_type
from ..utils.response_helpers import error_response

BASE_DIR = Path(__file__).parents[2]
UPLOAD_DIR = "uploads"

router = APIRouter(
    prefix='/process',
    tags=['process_data']
)

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
):
    try:
        ext = name.split(".")[-1]
        client_ip = request.client.host
        tmp_filename = f"tmp_{hashlib.md5((name + client_ip).encode()).hexdigest()}.{ext}"
        tmp_filepath = os.path.join(BASE_DIR, UPLOAD_DIR, tmp_filename)
        data = await file.read()

        if chunk_number == 0 and os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)

        with open(tmp_filepath, "ab") as buffer:
            buffer.write(data)

        if chunk_number == total_chunks - 1:
            final_filename = f"{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}.{ext}"
            final_filepath = os.path.join(BASE_DIR, UPLOAD_DIR, final_filename)
            os.rename(tmp_filepath, final_filepath)
            return JSONResponse({"finalFilename": final_filename})

        return JSONResponse({"status": "ok"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/process_data')
async def process_data(
    file_name: str,
    brightness: str = 50,
    sharpness: str = 50,
    contrast: str = 50
):
    if not file_name:
        return error_response(
            error_type="File Missing",
            details="File name not provided",
            message="File cannot be processed",
            code=500
        )

    try:
        file_path = os.path.join(BASE_DIR, UPLOAD_DIR, file_name)
        file_type = check_file_type(file_name)

        if file_type == 'image':
            encoded_image = await process_image(file_path, brightness, sharpness, contrast)
            return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")

        elif file_type == 'video':
            start_time = time.time()

            if check_resolution(file_path):
                print(f"Video resolution exceeds 1280x720: Skipping compression.")
                output_file = os.path.join(BASE_DIR, UPLOAD_DIR, f"bw_{file_name}")
                print("Generating output video")
                out_bw_video_path = await deoldify_video(
                    brightness=brightness,
                    sharpness=sharpness,
                    contrast=contrast,
                    input_video_path=file_path,
                    output_video_path=output_file
                )
            else:
                print(f"Running compression step")
                compressed_file = os.path.join(BASE_DIR, UPLOAD_DIR, f"compressed_{file_name}")
                compressed_video_path = await compress_video(input_video_path=file_path,
                                                            output_video_path=compressed_file)
                output_file = os.path.join(BASE_DIR, UPLOAD_DIR, f"bw_{file_name}")
                print("Generating output video")
                out_bw_video_path = await deoldify_video(brightness=brightness,
                                                    sharpness=sharpness,
                                                    contrast=contrast,
                                                    input_video_path=compressed_video_path,
                                                    output_video_path=output_file)
                os.remove(compressed_video_path)

            print(f"Total time taken - {time.time() - start_time}")

            # iterator for chunking video file
            def iterfile():
                with open(out_bw_video_path, "rb") as file_data:
                    yield from file_data

            print("Streaming video response")
            return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(e)
        return error_response(
            error_type="Internal Server Error",
            details="Some Interval Server Error Occured",
            message="Server could not complete the request",
            code=500
        )