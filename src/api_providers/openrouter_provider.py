from typing import Tuple, Optional, List
import os
import time
import base64
import requests
from pydub import AudioSegment
from openai import OpenAI, OpenAIError, NotFoundError

from .base_provider import BaseAudioProvider, BaseTextProvider
from prompts import PromptTemplate

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_STT_MODELS = [
    "openai/whisper-1",
    "openai/gpt-4o-mini-transcribe",
    "openai/gpt-4o-transcribe",
]

DEFAULT_CHAT_AUDIO_MODELS = [
    "google/gemini-2.5-flash",
    "openai/gpt-4o-audio-preview",
]

DEFAULT_CHAT_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "openrouter/auto",
]


def _fetch_models(api_key: str, **params) -> List[str]:
    """Fetch model ids from OpenRouter's /models endpoint with optional filters"""
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return sorted(model["id"] for model in response.json().get("data", []))


class OpenRouterAudioProvider(BaseAudioProvider):
    """OpenRouter implementation of the audio provider.

    Supports two transcription modes:
      - "stt": OpenRouter's dedicated /audio/transcriptions endpoint
        (OpenAI SDK compatible, works with Whisper-class and token-priced
        STT models)
      - "chat_audio": sends the audio as input_audio content to a
        multimodal chat completions model, useful when conversational
        analysis of the audio (not just verbatim transcription) is wanted
    """

    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter audio provider

        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def downsample_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Downsample audio to 16kHz mono (required by most transcription models)

        Args:
            audio_segment: Audio segment to downsample

        Returns:
            Downsampled audio segment
        """
        return audio_segment.set_frame_rate(16000).set_channels(1)

    def transcribe_file(self, file_path: str, model: str, mode: str = "stt") -> Tuple[str, bool]:
        """
        Transcribe an audio file via OpenRouter

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            mode: "stt" (dedicated transcription endpoint) or "chat_audio"
                  (multimodal chat completions model)

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        temp_path = ""
        try:
            audio = AudioSegment.from_file(file_path)
            audio = self.downsample_audio(audio)

            temp_path = file_path + '_optimized.wav'
            audio.export(temp_path, format='wav')

            if mode == "chat_audio":
                return self._transcribe_via_chat(temp_path, model)
            return self._transcribe_via_stt_endpoint(temp_path, model)

        except Exception as e:
            return f"Transcription error: {str(e)}", False

        finally:
            # Delete temporary file with retries
            if temp_path and os.path.exists(temp_path):
                max_retries = 3
                for i in range(max_retries):
                    try:
                        os.unlink(temp_path)
                        break
                    except PermissionError:
                        if i < max_retries - 1:  # Don't wait on last attempt
                            time.sleep(0.1 * (i + 1))

    def _transcribe_via_stt_endpoint(self, temp_path: str, model: str) -> Tuple[str, bool]:
        """Transcribe using OpenRouter's dedicated /audio/transcriptions endpoint"""
        with open(temp_path, 'rb') as f:
            try:
                transcription = self.client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    language="de",
                    prompt="This is a recording of a German speaker.",
                )
                return transcription.text, True
            except NotFoundError:
                print(f"Model {model} not found, using default model 'openai/whisper-1'.")
                f.seek(0)
                transcription = self.client.audio.transcriptions.create(
                    model="openai/whisper-1",
                    file=f,
                    language="de",
                    prompt="This is a recording of a German speaker.",
                )
                return transcription.text, True
            except OpenAIError as e:
                return f"Transcription error: {str(e)}", False

    def _transcribe_via_chat(self, temp_path: str, model: str) -> Tuple[str, bool]:
        """Transcribe by sending the audio as input_audio to a multimodal chat model"""
        with open(temp_path, 'rb') as f:
            base64_audio = base64.b64encode(f.read()).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe this German audio recording verbatim. Return only the transcript, no commentary.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content, True
        except OpenAIError as e:
            return f"Transcription error: {str(e)}", False

    def get_available_transcription_models(self, mode: str = "stt") -> List[str]:
        """
        Get available transcription models from OpenRouter

        Args:
            mode: "stt" lists models exposing the "transcription" output
                  modality, "chat_audio" lists chat models that accept
                  "audio" as an input modality

        Returns:
            List of available model names
        """
        try:
            if mode == "chat_audio":
                models = _fetch_models(self.api_key, input_modalities="audio")
            else:
                models = _fetch_models(self.api_key, output_modalities="transcription")

            if models:
                return models
        except Exception as e:
            print(f"Error fetching OpenRouter transcription models: {str(e)}")

        return DEFAULT_CHAT_AUDIO_MODELS if mode == "chat_audio" else DEFAULT_STT_MODELS


class OpenRouterTextProvider(BaseTextProvider):
    """OpenRouter implementation of the text provider"""

    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter text provider

        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL
        )

    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """
        Process text using OpenRouter's API

        Args:
            text: Text to process
            prompt_template: Prompt template to use
            model: Model to use for processing (optional)
            temperature: Temperature parameter for generation (optional)

        Returns:
            Processed text or None if processing failed
        """
        try:
            # Default model if none provided
            model_name = model if model else "openai/gpt-4o-mini"

            # Add provider preferences for better routing
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_template.system_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=temperature,
                extra_body={
                    "provider": {
                        "allow_fallbacks": True
                    }
                }
            )
            return response.choices[0].message.content

        except OpenAIError as e:
            error_message = f"Error during text processing: {e.type}"
            print(error_message)

            if e.type == "not_found":
                return f"Error: Model '{model_name}' not found. Please select a different model or use 'openrouter/auto' for automatic routing."
            elif e.type == "invalid_request_error":
                # Get more details from the error
                error_details = str(e)
                if "maximum context length" in error_details.lower():
                    return "Error: Text is too long for this model. Please use a shorter text or select a model with larger context window."
                elif "rate limit" in error_details.lower():
                    return "Error: Rate limit exceeded. Please wait a moment before trying again."
                else:
                    return f"Error: Invalid request - {error_details}"
            elif e.type == "api_connection_error":
                return "Error: Connection to OpenRouter API failed. Please check your internet connection."
            else:
                return f"Error: An unknown error occurred - {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

    def get_available_chat_models(self) -> List[str]:
        """
        Get available chat models for OpenRouter

        Returns:
            List of available model names
        """
        try:
            models = _fetch_models(self.api_key, output_modalities="text")

            if models:
                # Add special routing option
                return models + ["openrouter/auto"]
        except Exception as e:
            print(f"Error fetching OpenRouter chat models: {str(e)}")

        return DEFAULT_CHAT_MODELS
