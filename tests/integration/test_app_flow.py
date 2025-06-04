"""
Integration tests for the overall application flow
"""
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from src.speech_to_text import AudioTranscriber
from src.text_processors import TextProcessor
from src.app import TranscriptionApp
from src.prompts import PromptTemplate

# Mark this module as containing integration tests
pytestmark = pytest.mark.integration

class TestAppFlow:
    """Integration tests for the application flow"""
    
    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing"""
        monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
        monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit functionality"""
        with patch('src.app.st') as mock_st:
            # Setup session state as a simple dictionary
            mock_st.session_state = {}
            yield mock_st
    
    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write some dummy data to the file
            temp_file.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            temp_file_path = temp_file.name
        
        yield temp_file_path
        
        # Clean up the file after the test
        os.unlink(temp_file_path)
    
    def test_transcription_flow(self, mock_env_vars, mock_streamlit, temp_audio_file, mock_provider_response):
        """Test the end-to-end transcription flow"""
        # Arrange
        with patch('src.app.AudioTranscriber') as MockTranscriber, \
             patch('src.app.TextProcessor') as MockTextProcessor, \
             patch('src.app.get_mcp_client') as mock_get_client:
            
            # Mock the transcriber
            mock_transcriber_instance = MagicMock()
            mock_transcriber_instance.transcribe_file.return_value = ("Transcription result", True)
            mock_transcriber_instance.get_available_models.return_value = ["whisper-large-v3"]
            MockTranscriber.return_value = mock_transcriber_instance
            
            # Mock the text processor
            mock_processor_instance = MagicMock()
            mock_processor_instance.process_text.return_value = "Processed text result"
            mock_processor_instance.get_available_models.return_value = ["llama-3.3-70b-versatile"]
            MockTextProcessor.return_value = mock_processor_instance
            
            # Mock MCP client
            mock_mcp_client = MagicMock()
            mock_get_client.return_value = mock_mcp_client
            
            # Act
            app = TranscriptionApp()
            
            # Mock the handle_file_upload method directly since we can't fully test the Streamlit UI
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(b'test audio content')
                tmp_file_path = tmp_file.name
            
            # Create a mock file uploader result
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.getvalue.return_value = b'test audio content'
            
            # Create a mock prompt template
            mock_prompt = PromptTemplate(
                name="Test Prompt", 
                description="Test description",
                system_prompt="Test system prompt"
            )
            
            # Call handle_file_upload method
            app.handle_file_upload(
                mock_uploaded_file, 
                "whisper-large-v3", 
                "llama-3.3-70b-versatile", 
                mock_prompt
            )
            
            # Assert
            mock_transcriber_instance.transcribe_file.assert_called_once()
            mock_processor_instance.process_text.assert_called_once_with(
                "Transcription result", 
                mock_prompt, 
                model="llama-3.3-70b-versatile"
            )
            
            # Clean up
            os.unlink(tmp_file_path)
    
    def test_caching_mechanism(self, mock_env_vars, mock_streamlit):
        """Test that caching works correctly"""
        # Arrange
        with patch('src.app.AudioTranscriber') as MockTranscriber, \
             patch('src.app.TextProcessor') as MockTextProcessor, \
             patch('src.app.get_mcp_client') as mock_get_client:
            
            # Mock the transcriber and processor
            mock_transcriber_instance = MagicMock()
            mock_processor_instance = MagicMock()
            MockTranscriber.return_value = mock_transcriber_instance
            MockTextProcessor.return_value = mock_processor_instance
            
            # Mock MCP client
            mock_mcp_client = MagicMock()
            mock_get_client.return_value = mock_mcp_client
            
            # Set up session state
            mock_streamlit.session_state.provider = "groq"
            mock_streamlit.session_state.transcription_cache = {}
            mock_streamlit.session_state.processed_text_cache = {}
            mock_streamlit.session_state.agent_messages = []
            mock_streamlit.session_state.mcp_connected = False
            mock_streamlit.session_state.cached_models = {
                "groq": {
                    "chat": ["llama-3.3-70b-versatile"],
                    "transcription": ["whisper-large-v3"]
                }
            }
            
            # Act
            app = TranscriptionApp()
            
            # Assert
            assert hasattr(app, 'transcriber')
            assert hasattr(app, 'text_processor')
            
            # Check that caching is set up
            assert 'transcription_cache' in mock_streamlit.session_state
            assert 'processed_text_cache' in mock_streamlit.session_state