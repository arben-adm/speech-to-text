from pydantic import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    MAX_AUDIO_SIZE: int = 25 * 1024 * 1024  # 25MB
    SUPPORTED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "m4a"]
    DEFAULT_SAMPLE_RATE: int = 16000
    DEFAULT_CHANNELS: int = 1
    
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_TRANSCRIPTION_MODEL: str = "openai/whisper-1"
    DEFAULT_CHAT_AUDIO_MODEL: str = "google/gemini-2.5-flash"
    DEFAULT_CHAT_MODEL: str = "openai/gpt-4o-mini"

    class Config:
        env_file = ".env"