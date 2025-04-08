import pytest
from src.app import TranscriptionApp
import streamlit as st

@pytest.mark.integration
def test_full_transcription_flow():
    app = TranscriptionApp()
    
    # Test file upload flow
    with open("tests/fixtures/test_audio.wav", "rb") as audio_file:
        result = app.handle_file_upload(
            audio_file,
            "whisper-large-v3",
            "gpt-4",
            "professional"
        )
        
    assert result is not None
    assert len(result) > 0