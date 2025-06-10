import cv2
import numpy as np


def adjust_image_properties(image, brightness: int, sharpness: int, contrast: int):
    # Adjust brightness
    if brightness != 50:
        image = cv2.convertScaleAbs(image, alpha=(brightness/50), beta=((2.54*brightness)-127))

    # Adjust sharpness
    if sharpness != 50:
        image = adjustable_unsharp_mask(image=image, sharpness=sharpness)

    # Adjust contrast
    if contrast != 50:
        image = enhance_black_and_white(image, contrast=contrast)

    return image


def adjustable_unsharp_mask(image, kernel_size = (5, 5), sigma = 5.0, sharpness = 50, threshold = 0):
    amount = (sharpness / 50.0) - 1.0

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    sharpened = (1 + amount) * image - amount * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

def enhance_black_and_white(image, contrast=50):
    factor = 2 - (contrast/50)
    enhanced = image.astype(np.float32)
    enhanced[enhanced < 100] *= factor
    enhanced[enhanced >= 200] = 200 + factor * (enhanced[enhanced >= 200] - 200)

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced