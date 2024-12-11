import os
import time
import random
import argparse
from prompt_processing.safe_prompt_gen import SafePromptGenerator
from image_generation.safe_image_gen import SafeImageGenerator
from image_generation.clip_evaluate import CLIPImageTextSimilarity
from image_processing.fill_anything import process_image

def main(args):    
    unsafe_prompt = args.unsafe_prompt
    unsafe_word = args.unsafe_word
    substitution_pool_size = args.substitution_pool_size
    key = args.key
    sam_ckpt = args.sam_ckpt
    output_dir = args.output_dir
    
    if output_dir == "./output":
        os.makedirs(output_dir, exist_ok=True)
    
    safePromptGenerator = SafePromptGenerator(key, substitution_pool_size)
    safeImageGenerator = SafeImageGenerator(key)
    evaluator = CLIPImageTextSimilarity()
    
    safe_prompts, substitution_pool = safePromptGenerator.get_safe_prompts(unsafe_prompt, unsafe_word)

    safe_images = safeImageGenerator.generate_safe_images(safe_prompts)

    unsafe_images = []
    
    for idx, safe_image in enumerate(safe_images):
        if(safe_image == None):
            unsafe_image = None
            continue
        target_text = substitution_pool[idx]
        text_prompt = unsafe_word
        unsafe_image = process_image(
            safe_image, target_text, text_prompt, sam_ckpt, 
        )
        unsafe_images.append(unsafe_image)
    
    result_image = evaluator.find_most_similar_image(unsafe_images, unsafe_prompt)
    save_path = os.path.join(output_dir, "result_image.png")
    result_image.save(save_path)
    print(f"Result image saved to {save_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--unsafe_prompt', 
        type=str, 
        required=True, 
        help="unsafe prompt"
    )
    parser.add_argument(
        '--unsafe_word', 
        type=str, 
        required=True, 
        help="unsafe word"
    )
    parser.add_argument(
        '--substitution_pool_size', 
        type=int, 
        required=True, 
        help="size of the substitution pool"
    )
    parser.add_argument(
        '--key', 
        type=str, 
        required=True, 
        help="API key"
    )
    parser.add_argument(
        '--sam_ckpt', 
        type=str, 
        required=True, 
        help="Path to the SAM"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="./output",
        help="save path"
    )

    args = parser.parse_args()
    main(args)