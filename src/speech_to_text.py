from typing import Tuple, List
from api_providers.provider_factory import ProviderFactory
from api_providers.base_provider import BaseAudioProvider

class AudioTranscriber:
    def __init__(self, provider: str, api_key: str):
        """
        Initialize AudioTranscriber with chosen provider

        Args:
            provider: Provider name ('openai' or 'groq')
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.audio_provider = ProviderFactory.get_audio_provider(provider, api_key)

    def transcribe_file(self, file_path: str, model: str) -> Tuple[str, bool]:
        """
        Transcribe an audio file using the selected provider

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        return self.audio_provider.transcribe_file(file_path, model)

    def get_available_models(self) -> List[str]:
        """
        Get available transcription models for the current provider

        Returns:
            List of available model names
        """
        return self.audio_provider.get_available_transcription_models()
