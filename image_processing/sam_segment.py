import numpy as np
from typing import List
from .segment_anything import SamPredictor, sam_model_registry


def predict_masks_with_sam(
    img: np.ndarray,
    point_coords: List[List[float]],
    point_labels: List[int],
    model_type: str,
    ckpt_p: str,
    device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    return masks, scores, logits