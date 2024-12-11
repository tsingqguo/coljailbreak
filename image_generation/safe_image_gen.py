import httpx
from io import BytesIO
from PIL import Image
import requests
from openai import OpenAI

class SafeImageGenerator:
    def __init__(self, key):
        """
        Initialize
        """
        self.client = OpenAI(api_key=key)

    def generate_safe_images(self, safe_prompts):
        """
        Generate images for a list of safe prompts using the OpenAI API.
        """
        images = []  
        for prompt in safe_prompts:
            try:
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )

                image_url = response.data[0].url
                print(f"Generated image URL for prompt '{prompt}': {image_url}")
                
                image = self.download_image_to_memory(image_url)
                images.append(image)
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                images.append(None)

        return images

    def download_image_to_memory(self, url):
        """
        Download an image from a URL into memory and return it as a PIL image.
        """
        response = requests.get(url)
        response.raise_for_status()  

        img = Image.open(BytesIO(response.content))
        return img