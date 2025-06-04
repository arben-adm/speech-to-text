"""
Configuration for pytest
"""
import os
import sys
import pytest
from dotenv import load_dotenv

# Add the src directory to the path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def mock_api_key():
    """Fixture that provides a mock API key for testing"""
    return "test-api-key-123456789"

@pytest.fixture
def mock_audio_file():
    """Fixture that provides a path to a test audio file"""
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'test_audio.wav')
    
@pytest.fixture
def sample_text():
    """Fixture that provides sample text for testing"""
    return "This is a sample text for testing text processing functionality."

@pytest.fixture
def mock_provider_response():
    """Fixture that provides a mock API response for testing"""
    return {
        "text": "This is a sample transcription result.",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "This is a sample",
                "tokens": [50364, 394, 309, 264, 2219],
                "temperature": 0.0,
                "avg_logprob": -0.45,
                "compression_ratio": 0.65,
                "no_speech_prob": 0.1
            },
            {
                "id": 1,
                "seek": 2000,
                "start": 2.0,
                "end": 4.0,
                "text": "transcription result.",
                "tokens": [5893, 3598, 13],
                "temperature": 0.0,
                "avg_logprob": -0.35,
                "compression_ratio": 0.7,
                "no_speech_prob": 0.05
            }
        ]
    }