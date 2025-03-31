from typing import Dict, Type, Union
from .base_provider import BaseAudioProvider, BaseTextProvider
from .openai_provider import OpenAIAudioProvider, OpenAITextProvider
from .groq_provider import GroqAudioProvider, GroqTextProvider
from .openrouter_provider import OpenRouterAudioProvider, OpenRouterTextProvider

class ProviderFactory:
    """Factory class to create provider instances"""
    
    # Registry of available audio providers
    _audio_providers: Dict[str, Type[BaseAudioProvider]] = {
        'openai': OpenAIAudioProvider,
        'groq': GroqAudioProvider,
        'openrouter': OpenRouterAudioProvider
    }
    
    # Registry of available text providers
    _text_providers: Dict[str, Type[BaseTextProvider]] = {
        'openai': OpenAITextProvider,
        'groq': GroqTextProvider,
        'openrouter': OpenRouterTextProvider
    }
    
    @classmethod
    def get_audio_provider(cls, provider_name: str, api_key: str) -> BaseAudioProvider:
        """
        Get an audio provider instance
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'groq')
            api_key: API key for the provider
            
        Returns:
            Instance of BaseAudioProvider
            
        Raises:
            ValueError: If provider_name is not supported
        """
        provider_class = cls._audio_providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unsupported audio provider: {provider_name}")
        
        return provider_class(api_key)
    
    @classmethod
    def get_text_provider(cls, provider_name: str, api_key: str) -> BaseTextProvider:
        """
        Get a text provider instance
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'groq')
            api_key: API key for the provider
            
        Returns:
            Instance of BaseTextProvider
            
        Raises:
            ValueError: If provider_name is not supported
        """
        provider_class = cls._text_providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unsupported text provider: {provider_name}")
        
        return provider_class(api_key)
    
    @classmethod
    def register_audio_provider(cls, name: str, provider_class: Type[BaseAudioProvider]) -> None:
        """
        Register a new audio provider
        
        Args:
            name: Name of the provider
            provider_class: Provider class
        """
        cls._audio_providers[name.lower()] = provider_class
    
    @classmethod
    def register_text_provider(cls, name: str, provider_class: Type[BaseTextProvider]) -> None:
        """
        Register a new text provider
        
        Args:
            name: Name of the provider
            provider_class: Provider class
        """
        cls._text_providers[name.lower()] = provider_class
