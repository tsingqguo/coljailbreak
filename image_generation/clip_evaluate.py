import clip
import torch
from PIL import Image

class CLIPImageTextSimilarity:
    def __init__(self):
        """
        Initialize
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def compute_similarity(self, image, text):
        """
        Compute the CLIP similarity between a given image and text.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text]).to(self.device)  
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        similarity = torch.cosine_similarity(image_features, text_features).cpu().numpy()[0]
        return similarity

    def find_most_similar_image(self, images, prompt):
        """
        Find the most similar image to a given text prompt from a list of images.
        """
        highest_similarity = -1  
        most_similar_image = None  

        for image in images:
            similarity = self.compute_similarity(image, prompt)
                        
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_image = image

        return most_similar_image