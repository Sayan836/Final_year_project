import cv2
from fastapi import HTTPException

from .adjust_properties import adjust_image_properties


async def process_image(file_path: str, brightness: str, sharpness: str, contrast: str):
    """
    Converts an image to black and white.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        changed_gray_image = adjust_image_properties(gray_image,
                                                     brightness=int(brightness),
                                                     sharpness=int(sharpness),
                                                     contrast=int(contrast))

        _, encoded_image = cv2.imencode(".png", changed_gray_image)

        return encoded_image
    except Exception as e:
        print(e)
