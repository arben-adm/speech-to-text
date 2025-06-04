"""
Unit tests for the AudioTranscriber class
"""
import pytest
from unittest.mock import patch, MagicMock
from src.speech_to_text import AudioTranscriber
from src.api_providers.base_provider import BaseAudioProvider

class TestAudioTranscriber:
    """Test cases for the AudioTranscriber class"""

    def test_initialization(self, mock_api_key):
        """Test that the transcriber initializes correctly"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='groq', api_key=mock_api_key)
            
            # Assert
            assert transcriber.provider == 'groq'
            assert transcriber.audio_provider == mock_provider
            mock_factory.assert_called_once_with('groq', mock_api_key)
    
    def test_transcribe_file_success(self, mock_api_key):
        """Test successful file transcription"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.transcribe_file.return_value = ("Transcription result", True)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='groq', api_key=mock_api_key)
            result, success = transcriber.transcribe_file("dummy_path.wav", "whisper-large-v3")
            
            # Assert
            assert result == "Transcription result"
            assert success is True
            mock_provider.transcribe_file.assert_called_once_with("dummy_path.wav", "whisper-large-v3")
    
    def test_transcribe_file_failure(self, mock_api_key):
        """Test failed file transcription"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.transcribe_file.return_value = ("Error message", False)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='groq', api_key=mock_api_key)
            result, success = transcriber.transcribe_file("dummy_path.wav", "whisper-large-v3")
            
            # Assert
            assert result == "Error message"
            assert success is False
            mock_provider.transcribe_file.assert_called_once_with("dummy_path.wav", "whisper-large-v3")
    
    def test_get_available_models(self, mock_api_key):
        """Test getting available transcription models"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.get_available_transcription_models.return_value = ["model1", "model2"]
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='groq', api_key=mock_api_key)
            models = transcriber.get_available_models()
            
            # Assert
            assert models == ["model1", "model2"]
            mock_provider.get_available_transcription_models.assert_called_once()