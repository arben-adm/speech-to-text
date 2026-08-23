"""Application-wide constants for audio transcription.

Plain module-level constants rather than a pydantic BaseSettings class: the
previous version imported `BaseSettings` from `pydantic`, which was removed
in Pydantic v2 (it now lives in the separate `pydantic-settings` package,
which was never a declared dependency), so this module has been dead code.
None of these values are read from the environment, so a settings framework
buys nothing here - a plain module is the single source of truth with zero
extra dependencies.
"""
from typing import List

MAX_AUDIO_SIZE: int = 25 * 1024 * 1024  # 25MB - OpenRouter/OpenAI transcription endpoint cap
SUPPORTED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "m4a"]
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_CHANNELS: int = 1

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_TRANSCRIPTION_MODEL: str = "mistralai/voxtral-small-24b-2507-stt"
DEFAULT_CHAT_AUDIO_MODEL: str = "google/gemini-2.5-flash"
DEFAULT_CHAT_MODEL: str = "openai/gpt-5.6-sol"
