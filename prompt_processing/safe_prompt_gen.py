import os
import httpx
from openai import OpenAI

class SafePromptGenerator:
    def __init__(self, key, substitution_pool_size):
        """
        Initialize
        """
        self.key = key
        self.substitution_pool_size = substitution_pool_size
        self.client = OpenAI(api_key=key)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_path = os.path.join(current_dir, 'template', 'template.txt')

    def load_template(self):
        """
        Load and return the contents of the template file.
        """
        with open(self.template_path, 'r', encoding='utf-8') as file:
            template_text = file.read()
        return template_text

    def generate_substitution_pool(self, unsafe_word):
        """
        Generate a pool of substitution words using the API.
        """
        template_text = self.load_template()
        textprompt = template_text.format(unsafe_word, self.substitution_pool_size)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": textprompt
                }
            ]
        )

        text = response.choices[0].message.content
        substitution_pool = [item.strip() for item in text.split(',')]
        return substitution_pool

    def get_safe_prompts(self, unsafe_prompt, unsafe_word):
        """
        Generate multiple safe prompts by replacing the unsafe word with words from the substitution pool.
        """

        substitution_pool = self.generate_substitution_pool(unsafe_word)
        safe_prompts = []
        
        for word in substitution_pool:
            safe_prompt = unsafe_prompt.replace(unsafe_word, word)
            safe_prompts.append(safe_prompt)
        
        return safe_prompts, substitution_pool