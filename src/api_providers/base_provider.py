from abc import ABC, abstractmethod
from typing import Tuple, Optional
from prompts import PromptTemplate

class BaseAudioProvider(ABC):
    """Abstract base class for audio transcription providers"""
    
    @abstractmethod
    def transcribe_file(self, file_path: str, model: str) -> Tuple[str, bool]:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            
        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        pass
    
    @abstractmethod
    def get_available_transcription_models(self) -> list[str]:
        """
        Get available transcription models for this provider
        
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
