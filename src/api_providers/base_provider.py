from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from prompts import PromptTemplate

class APIError(Exception):
    def __init__(self, message: str, provider: str, error_type: str):
        self.message = message
        self.provider = provider
        self.error_type = error_type
        super().__init__(self.message)

class BaseProvider:
    def handle_api_error(self, error: Exception, context: str) -> APIError:
        """Standardisierte Fehlerbehandlung für API-Aufrufe"""
        if isinstance(error, APIError):
            return error

        error_type = getattr(error, 'type', 'unknown')
        return APIError(
            message=f"{context}: {str(error)}",
            provider=self.__class__.__name__,
            error_type=error_type
        )

class BaseAudioProvider(ABC):
    """Abstract base class for audio transcription providers"""

    @abstractmethod
    def transcribe_file(self, file_path: str, model: str, mode: str = "stt") -> Tuple[str, bool]:
        """
        Transcribe an audio file

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            mode: Transcription mode - "stt" for the dedicated transcription
                  endpoint, or "chat_audio" to send the audio as input to a
                  multimodal chat completions model

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        pass

    @abstractmethod
    def get_available_transcription_models(self, mode: str = "stt") -> list[str]:
        """
        Get available transcription models for this provider

        Args:
            mode: "stt" for dedicated transcription models, "chat_audio" for
                  multimodal chat models that accept audio input

        Returns:
            List of available model names
        """
        pass


class BaseTextProvider(ABC):
    """Abstract base class for text processing providers"""

    @abstractmethod
    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """
        Process text using the provider's API

        Args:
            text: Text to process
            prompt_template: Prompt template to use
            model: Model to use for processing (optional)
            temperature: Temperature parameter for generation (optional)

        Returns:
            Processed text or None if processing failed
        """
        pass

    @abstractmethod
    def get_available_chat_models(self) -> list[str]:
        """
        Get available chat models for this provider

        Returns:
            List of available model names
        """
        pass
