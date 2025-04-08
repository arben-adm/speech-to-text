from pydantic import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    MAX_AUDIO_SIZE: int = 25 * 1024 * 1024  # 25MB
    SUPPORTED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "m4a"]
    DEFAULT_SAMPLE_RATE: int = 16000
    DEFAULT_CHANNELS: int = 1
    
    PROVIDER_CONFIGS: Dict[str, Dict] = {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "whisper-large-v3"
        },
        "openai": {
            "default_model": "whisper-1"
        }
    }
    
    class Config:
        env_file = ".env"