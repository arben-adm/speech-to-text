from typing import Tuple, List
from api_providers.provider_factory import ProviderFactory
from api_providers.base_provider import BaseAudioProvider

class AudioTranscriber:
    def __init__(self, provider: str, api_key: str):
        """
        Initialize AudioTranscriber with chosen provider

        Args:
            provider: Provider name (currently only 'openrouter')
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.audio_provider = ProviderFactory.get_audio_provider(provider, api_key)

    def transcribe_file(self, file_path: str, model: str, mode: str = "stt", language: str | None = "de") -> Tuple[str, bool]:
        """
        Transcribe an audio file using the selected provider

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            mode: "stt" (dedicated transcription endpoint) or "chat_audio"
                  (multimodal chat completions model)
            language: ISO-639-1 language code (e.g. "de"), or None to let
                      the model auto-detect the spoken language

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        return self.audio_provider.transcribe_file(file_path, model, mode=mode, language=language)

    def get_available_models(self, mode: str = "stt") -> List[str]:
        """
        Get available transcription models for the current provider

        Args:
            mode: "stt" or "chat_audio", see transcribe_file

        Returns:
            List of available model names
        """
        return self.audio_provider.get_available_transcription_models(mode=mode)
