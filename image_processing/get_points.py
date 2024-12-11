import cv2
import torch
import numpy as np
from .clip import clip
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("CS-ViT-B/16", device=device)
model.eval()

preprocess = Compose([Resize((512, 512), interpolation=BICUBIC), 
                      ToTensor(),
                      Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))])

def get_clip_surgery_result(image, target_text, threshold=0.8):
    pil_img = image
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image = preprocess(pil_img).unsqueeze(0).to(device)

    texts = [target_text]

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)

        redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)

        similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0]

        points, labels = clip.similarity_map_to_points(similarity[1:, 0], cv2_img.shape[:2], t=threshold)

        points_np = np.array(points)
        labels_np = np.array(labels)

        selected_points = points_np[labels_np == 1]

        if len(selected_points) > 0:
            mean_x = np.mean(selected_points[:, 0])
            mean_y = np.mean(selected_points[:, 1])
            result = [mean_x, mean_y]
        else:
            result = None

        return result