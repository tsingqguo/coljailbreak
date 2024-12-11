import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List


def load_img_to_array(img_p):
    if isinstance(img_p, Image.Image):
        img = img_p
    else:
        img = Image.open(img_p)  
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask