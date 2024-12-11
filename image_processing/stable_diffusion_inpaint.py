import cv2
import torch
import numpy as np
import PIL.Image as Image
from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline 
from .utils.mask_processing import crop_for_filling_pre, crop_for_filling_post


def fill_img_with_sd(
    img: np.ndarray,
    mask: np.ndarray,
    text_prompt: str,
    device="cuda"
):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "Vijish/fooocus_inpainting", torch_dtype=torch.float16
    ).to(device)
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    
    img_crop_filled = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    
    crop_height, crop_width = img_crop.shape[:2] 
    img_crop_filled = cv2.resize(np.array(img_crop_filled), (crop_width, crop_height))

    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled