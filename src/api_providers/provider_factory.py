from typing import Dict, Type
from .base_provider import BaseAudioProvider, BaseTextProvider
from .openrouter_provider import OpenRouterAudioProvider, OpenRouterTextProvider

class ProviderFactory:
    """Factory class to create provider instances. OpenRouter is the sole
    supported provider - it routes to whichever underlying model/provider
    is requested via the model slug, so no other provider integration is
    needed."""

    _audio_providers: Dict[str, Type[BaseAudioProvider]] = {
        'openrouter': OpenRouterAudioProvider
    }

    _text_providers: Dict[str, Type[BaseTextProvider]] = {
        'openrouter': OpenRouterTextProvider
    }

    @classmethod
    def get_audio_provider(cls, provider_name: str, api_key: str) -> BaseAudioProvider:
        """
        Get an audio provider instance

        Args:
            provider_name: Name of the provider (currently only 'openrouter')
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
            provider_name: Name of the provider (currently only 'openrouter')
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
