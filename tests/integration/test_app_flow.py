import pytest
import os
from unittest.mock import MagicMock, patch
from src.app import TranscriptionApp
import streamlit as st

@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists("tests/fixtures/test_audio.wav"),
                    reason="Test audio file not found")
def test_full_transcription_flow():
    # Skip this test for now until we have proper fixtures
    pytest.skip("Integration test requires proper fixtures and mocked Streamlit")

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