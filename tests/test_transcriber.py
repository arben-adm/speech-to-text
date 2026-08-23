"""
Unit tests for the AudioTranscriber class
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from pydub import AudioSegment
from src.speech_to_text import AudioTranscriber
from src.api_providers.base_provider import BaseAudioProvider
from src.api_providers.openrouter_provider import OpenRouterAudioProvider
from src.api_providers import openrouter_provider as openrouter_provider_module

class TestAudioTranscriber:
    """Test cases for the AudioTranscriber class"""

    def test_initialization(self, mock_api_key):
        """Test that the transcriber initializes correctly"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='openrouter', api_key=mock_api_key)
            
            # Assert
            assert transcriber.provider == 'openrouter'
            assert transcriber.audio_provider == mock_provider
            mock_factory.assert_called_once_with('openrouter', mock_api_key)
    
    def test_transcribe_file_success(self, mock_api_key):
        """Test successful file transcription"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.transcribe_file.return_value = ("Transcription result", True)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='openrouter', api_key=mock_api_key)
            result, success = transcriber.transcribe_file("dummy_path.wav", "whisper-large-v3")
            
            # Assert
            assert result == "Transcription result"
            assert success is True
            mock_provider.transcribe_file.assert_called_once_with("dummy_path.wav", "whisper-large-v3", mode="stt", language="de")
    
    def test_transcribe_file_failure(self, mock_api_key):
        """Test failed file transcription"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.transcribe_file.return_value = ("Error message", False)
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='openrouter', api_key=mock_api_key)
            result, success = transcriber.transcribe_file("dummy_path.wav", "whisper-large-v3")
            
            # Assert
            assert result == "Error message"
            assert success is False
            mock_provider.transcribe_file.assert_called_once_with("dummy_path.wav", "whisper-large-v3", mode="stt", language="de")
    
    def test_get_available_models(self, mock_api_key):
        """Test getting available transcription models"""
        with patch('src.speech_to_text.ProviderFactory.get_audio_provider') as mock_factory:
            # Arrange
            mock_provider = MagicMock(spec=BaseAudioProvider)
            mock_provider.get_available_transcription_models.return_value = ["model1", "model2"]
            mock_factory.return_value = mock_provider
            
            # Act
            transcriber = AudioTranscriber(provider='openrouter', api_key=mock_api_key)
            models = transcriber.get_available_models()
            
            # Assert
            assert models == ["model1", "model2"]
            mock_provider.get_available_transcription_models.assert_called_once()


class TestOpenRouterAudioProviderMP3:
    """Test cases for the MP3-first transcription path in OpenRouterAudioProvider"""

    def test_prepare_audio_keeps_mp3(self, mock_api_key, tone_mp3_path):
        """An MP3 upload should stay MP3 through _prepare_audio, not get re-exported to WAV"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)

        prepared_path, fmt = provider._prepare_audio(tone_mp3_path)
        try:
            assert fmt == "mp3"
            assert prepared_path.endswith(".mp3")
        finally:
            os.unlink(prepared_path)

    def test_chat_audio_format_matches_prepared_file(self, mock_api_key, tone_mp3_path):
        """chat_audio mode must send input_audio.format matching the prepared file, not a hardcoded 'wav'"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hallo"))]
        provider.client.chat.completions.create = MagicMock(return_value=mock_response)

        text, success = provider.transcribe_file(tone_mp3_path, "google/gemini-2.5-flash", mode="chat_audio")

        assert success is True
        assert text == "hallo"
        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        input_audio = call_kwargs["messages"][0]["content"][1]["input_audio"]
        assert input_audio["format"] == "mp3"

    def test_transcribe_file_chunks_when_over_limit(self, mock_api_key, tone_mp3_path, monkeypatch):
        """A prepared file over MAX_AUDIO_SIZE must be split and transcribed part by part, then joined"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)

        fake_chunks = [AudioSegment.silent(duration=500) for _ in range(3)]
        monkeypatch.setattr(provider, "_split_audio", lambda segment, max_bytes: fake_chunks)
        monkeypatch.setattr(openrouter_provider_module, "MAX_AUDIO_SIZE", 1)  # force the chunk path

        responses = [MagicMock(text=f"part{i}") for i in range(3)]
        mock_create = MagicMock(side_effect=responses)
        provider.client.audio.transcriptions.create = mock_create

        text, success = provider.transcribe_file(tone_mp3_path, "openai/whisper-1", mode="stt")

        assert success is True
        assert mock_create.call_count == 3
        assert text == "part0 part1 part2"

    def test_split_audio_snaps_to_silence(self, mock_api_key, tone_silence_wav_path):
        """Cut points must snap to a detected silence gap, not land at a raw byte/time offset mid-tone"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)
        segment = AudioSegment.from_file(tone_silence_wav_path)

        # The fixture is 8s tone + 2s true silence + 8s tone. max_bytes=76000 gives a
        # nominal (raw) cut at 9500ms, which sits inside the second tone - the actual
        # cut must snap back into the true silence gap at ~[7994, 10006]ms instead.
        chunks = provider._split_audio(segment, max_bytes=76_000)

        assert len(chunks) == 2
        first_chunk_ms = len(chunks[0])
        assert abs(first_chunk_ms - 9500) > 200
        assert 7900 <= first_chunk_ms <= 10100

    def test_language_none_omits_language_and_prompt(self, mock_api_key, tone_mp3_path):
        """language=None must omit both the `language` kwarg and the German-speaker prompt"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)
        mock_create = MagicMock(return_value=MagicMock(text="hi"))
        provider.client.audio.transcriptions.create = mock_create

        text, success = provider.transcribe_file(tone_mp3_path, "openai/whisper-1", mode="stt", language=None)

        assert success is True
        call_kwargs = mock_create.call_args.kwargs
        assert "language" not in call_kwargs
        assert "prompt" not in call_kwargs

    def test_default_language_includes_german_prompt(self, mock_api_key, tone_mp3_path):
        """The default language='de' must still send both `language` and the German-speaker prompt"""
        provider = OpenRouterAudioProvider(api_key=mock_api_key)
        mock_create = MagicMock(return_value=MagicMock(text="hi"))
        provider.client.audio.transcriptions.create = mock_create

        provider.transcribe_file(tone_mp3_path, "openai/whisper-1", mode="stt")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["language"] == "de"
        assert "German speaker" in call_kwargs["prompt"]