import torch
import numpy as np
from .sam_segment import predict_masks_with_sam
from .stable_diffusion_inpaint import fill_img_with_sd
from .utils import load_img_to_array, dilate_mask
from .get_points import get_clip_surgery_result
from PIL import Image

def process_image(
    input_img, target_text, text_prompt, sam_ckpt,
    point_labels=1, dilate_kernel_size=15, sam_model_type="vit_h", seed=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    latest_coords = get_clip_surgery_result(input_img, target_text)
    point_labels = torch.tensor([point_labels])
    
    img = load_img_to_array(input_img)
    
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    for i in range(1): 
        if dilate_kernel_size is not None:
            masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    for idx, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed)
        
        img_filled = fill_img_with_sd(img, mask, text_prompt, device=device)
        if idx == 2:
            filled_image = Image.fromarray(img_filled)

    return filled_image