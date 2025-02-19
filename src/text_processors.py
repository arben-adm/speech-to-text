from typing import Optional
from openai import OpenAI, OpenAIError
from prompts import PromptTemplate

class TextProcessor:
    def __init__(self, provider: str, api_key: str):
        """Initialize TextProcessor with selected provider"""
        self.provider = provider.lower()
        self.base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )
    
    def get_model_name(self, model: str = None):
        """Returns the appropriate model based on the provider, default is Groq"""
        if model:
            return model
        elif self.provider == 'groq':
            return "llama-3.3-70b-versatile"
        return "gpt-4o-mini"
    
    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """Process text with selected provider and prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.get_model_name(model),
                messages=[
                    {
                        "role": "system",
                        "content": prompt_template.system_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"Error during text processing: {e.type}")
            if e.type == "not_found":
                return "Error: Model not found. Please check the model name."
            elif e.type == "invalid_request_error":
                return "Error: Invalid request. Please check the parameters."
            elif e.type == "api_connection_error":
                return "Error: Connection to API server failed. Please check your internet connection."
            else:
                return "Error: An unknown error occurred."
