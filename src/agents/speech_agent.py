"""
Speech Agent - Integrates speech-to-text functionality with AI agents using MCP servers
"""
from typing import Dict, List, Any, Optional, Callable, Awaitable
import asyncio
import tempfile
import os

from agents.agent import Agent, Tool
from speech_to_text import AudioTranscriber
from text_processors import TextProcessor
from utils.logger import get_logger

logger = get_logger(__name__)

class TranscriptionTool(Tool):
    """Tool for transcribing audio files"""
    
    def __init__(self, transcriber: AudioTranscriber, default_model: str):
        super().__init__(
            name="transcribe",
            description="Transcribe an audio file to text"
        )
        self.transcriber = transcriber
        self.default_model = default_model
        
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the transcription
        
        Args:
            args: Dictionary containing:
                - file_path: Path to the audio file
                - model: Optional transcription model
                
        Returns:
            Dictionary with transcription result
        """
        file_path = args.get("file_path", "")
        model = args.get("model", self.default_model)
        
        if not file_path:
            return {
                "error": "No file path provided",
                "isError": True
            }
            
        try:
            text, success = self.transcriber.transcribe_file(file_path, model)
            
            if success:
                return {
                    "result": text,
                    "isError": False
                }
            else:
                return {
                    "error": text,  # Error message is returned in the text variable
                    "isError": True
                }
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return {
                "error": f"Error during transcription: {str(e)}",
                "isError": True
            }


class TextProcessingTool(Tool):
    """Tool for processing text using AI"""
    
    def __init__(self, text_processor: TextProcessor, default_model: str):
        super().__init__(
            name="process_text",
            description="Process text using AI"
        )
        self.text_processor = text_processor
        self.default_model = default_model
        
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute text processing
        
        Args:
            args: Dictionary containing:
                - text: Text to process
                - system_prompt: System prompt for the AI
                - model: Optional model name
                
        Returns:
            Dictionary with processed text result
        """
        text = args.get("text", "")
        system_prompt = args.get("system_prompt", "")
        model = args.get("model", self.default_model)
        
        if not text:
            return {
                "error": "No text provided",
                "isError": True
            }
            
        if not system_prompt:
            return {
                "error": "No system prompt provided",
                "isError": True
            }
            
        try:
            from src.prompts import PromptTemplate
            
            # Create a custom prompt template
            prompt = PromptTemplate(
                name="Custom",
                description="Custom prompt",
                system_prompt=system_prompt
            )
            
            processed_text = self.text_processor.process_text(
                text,
                prompt,
                model=model
            )
            
            return {
                "result": processed_text,
                "isError": False
            }
        except Exception as e:
            logger.error(f"Error during text processing: {str(e)}")
            return {
                "error": f"Error during text processing: {str(e)}",
                "isError": True
            }


class SpeechAgent(Agent):
    """
    AI Agent that combines speech-to-text functionality with MCP servers
    for advanced processing capabilities
    """
    
    def __init__(
        self,
        name: str,
        system: str,
        provider: str,
        api_key: str,
        transcription_model: str,
        chat_model: str,
        mcp_servers: List[Dict[str, Any]] = None,
        mcp_config_path: str = "mcp_config.json"
    ):
        """
        Initialize Speech Agent
        
        Args:
            name: Agent name
            system: System prompt for the agent
            provider: API provider name ('openai', 'groq', etc.)
            api_key: API key for the provider
            transcription_model: Default transcription model
            chat_model: Default chat model
            mcp_servers: List of MCP server configurations to add
            mcp_config_path: Path to MCP configuration file
        """
        # Initialize speech-to-text components
        self.transcriber = AudioTranscriber(provider=provider, api_key=api_key)
        self.text_processor = TextProcessor(provider=provider, api_key=api_key)
        
        # Create local tools
        tools = [
            TranscriptionTool(self.transcriber, transcription_model),
            TextProcessingTool(self.text_processor, chat_model)
        ]
        
        # Initialize the base Agent class
        super().__init__(
            name=name,
            system=system,
            tools=tools,
            mcp_servers=mcp_servers,
            mcp_config_path=mcp_config_path
        )
        
    async def transcribe_and_process(
        self,
        audio_bytes: bytes,
        transcription_model: str,
        chat_model: str,
        system_prompt: str,
        callback: Optional[Callable[[str, str, str], Awaitable[None]]] = None
    ) -> Dict[str, str]:
        """
        Transcribe audio and process the resulting text
        
        Args:
            audio_bytes: Audio data as bytes
            transcription_model: Model to use for transcription
            chat_model: Model to use for text processing
            system_prompt: System prompt for text processing
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with original and processed text
        """
        tmp_file_path = ""
        try:
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Transcribe the audio
            if callback:
                await callback("status", "Transcribing audio...", "")
                
            transcription_result = await self.execute_tool("transcribe", {
                "file_path": tmp_file_path,
                "model": transcription_model
            })
            
            if transcription_result.get("isError", False):
                if callback:
                    await callback("error", transcription_result.get("error", "Transcription failed"), "")
                return {
                    "error": transcription_result.get("error", "Transcription failed")
                }
            
            original_text = transcription_result.get("result", "")
            
            if callback:
                await callback("transcription", original_text, "")
                await callback("status", "Processing text...", "")
            
            # Process the transcribed text
            processing_result = await self.execute_tool("process_text", {
                "text": original_text,
                "system_prompt": system_prompt,
                "model": chat_model
            })
            
            if processing_result.get("isError", False):
                if callback:
                    await callback("error", processing_result.get("error", "Text processing failed"), "")
                return {
                    "original_text": original_text,
                    "error": processing_result.get("error", "Text processing failed")
                }
            
            processed_text = processing_result.get("result", "")
            
            if callback:
                await callback("processed", processed_text, "")
            
            return {
                "original_text": original_text,
                "processed_text": processed_text
            }
            
        except Exception as e:
            logger.error(f"Error in transcribe_and_process: {str(e)}")
            if callback:
                await callback("error", f"Error: {str(e)}", "")
            return {
                "error": f"Error: {str(e)}"
            }
            
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")