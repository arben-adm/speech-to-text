import pytest
from src.speech_to_text import AudioTranscriber
from unittest.mock import Mock, patch

def test_transcribe_file():
    mock_provider = Mock()
    mock_provider.transcribe_file.return_value = ("Test transcription", True)

    transcriber = AudioTranscriber("groq", "fake_key")
    transcriber.audio_provider = mock_provider

    result, success = transcriber.transcribe_file("test.wav", "whisper-large-v3")

    assert success == True
    assert result == "Test transcription"
    assert mock_provider.transcribe_file.call_count == 1