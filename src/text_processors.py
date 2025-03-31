from typing import Optional, List
from prompts import PromptTemplate
from api_providers.provider_factory import ProviderFactory
from api_providers.base_provider import BaseTextProvider

class TextProcessor:
    def __init__(self, provider: str, api_key: str):
        """
        Initialize TextProcessor with selected provider
        
        Args:
            provider: Provider name ('openai' or 'groq')
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.text_provider = ProviderFactory.get_text_provider(provider, api_key)
    
    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """
        Process text with selected provider and prompt
        
        Args:
            text: Text to process
            prompt_template: Prompt template to use
            model: Model to use for processing (optional)
            temperature: Temperature parameter for generation (optional)
            
        Returns:
            Processed text or None if processing failed
        """
        return self.text_provider.process_text(text, prompt_template, model, temperature)
    
    def get_available_models(self) -> List[str]:
        """
        Get available chat models for the current provider
        
        Returns:
            List of available model names
        """
        return self.text_provider.get_available_chat_models()
