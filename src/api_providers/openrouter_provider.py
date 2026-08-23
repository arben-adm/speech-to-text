from typing import Tuple, Optional, List
import os
import time
import base64
import requests
from pydub import AudioSegment
from pydub.silence import detect_silence
from openai import OpenAI, OpenAIError, NotFoundError

from .base_provider import BaseAudioProvider, BaseTextProvider
from prompts import PromptTemplate
from config.settings import (
    MAX_AUDIO_SIZE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_CHAT_AUDIO_MODEL,
    DEFAULT_CHAT_MODEL,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_STT_MODELS = [
    DEFAULT_TRANSCRIPTION_MODEL,
    "openai/whisper-1",
    "openai/gpt-4o-mini-transcribe",
    "openai/gpt-4o-transcribe",
]

DEFAULT_CHAT_AUDIO_MODELS = [
    DEFAULT_CHAT_AUDIO_MODEL,
    "openai/gpt-4o-audio-preview",
]

DEFAULT_CHAT_MODELS = [
    DEFAULT_CHAT_MODEL,
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "openrouter/auto",
]

# 64kbps mono MP3 is ~8KB/s, so the 25MB endpoint cap covers ~50 minutes of
# audio instead of the ~13 minutes a 16kHz mono WAV allows.
_MP3_BITRATE_KBPS = 64
_MP3_BITRATE = "64k"
_MAX_CHUNK_MS = 10 * 60 * 1000  # 10 minutes
_SILENCE_SEARCH_WINDOW_MS = 15 * 1000  # +/- 15s around each nominal cut point

GERMAN_SPEAKER_PROMPT = "This is a recording of a German speaker."


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

    Both accept mp3 as an input format (confirmed against OpenRouter's docs),
    so uploaded audio is downsampled and re-encoded as MP3 rather than WAV -
    this keeps files small enough to stay under the endpoint's 25MB cap for
    much longer recordings. Files that still exceed the cap are split into
    silence-snapped chunks and transcribed sequentially.
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
        return audio_segment.set_frame_rate(DEFAULT_SAMPLE_RATE).set_channels(1)

    def _prepare_audio(self, file_path: str) -> Tuple[str, str]:
        """
        Downsample audio to 16kHz mono and re-encode it as MP3 for upload.

        Args:
            file_path: Path to the source audio file

        Returns:
            Tuple of (prepared_file_path, format)
        """
        try:
            audio = AudioSegment.from_file(file_path)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg is required to read audio files but was not found on PATH. "
                "Please install ffmpeg."
            ) from e

        audio = self.downsample_audio(audio)

        base, _ = os.path.splitext(file_path)
        prepared_path = f"{base}_prepared.mp3"
        try:
            audio.export(prepared_path, format="mp3", bitrate=_MP3_BITRATE)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg is required to export audio as MP3 but was not found on PATH. "
                "Please install ffmpeg."
            ) from e

        return prepared_path, "mp3"

    def _split_audio(self, segment: AudioSegment, max_bytes: int) -> List[AudioSegment]:
        """
        Split an audio segment into chunks no longer than 10 minutes (or
        shorter, if max_bytes is tighter than that at the MP3 encoding
        bitrate). Each interior cut point is snapped to the nearest silence
        detected within a +/-15s window around the nominal cut, so words are
        not sliced mid-utterance.

        Args:
            segment: Audio segment to split (already downsampled)
            max_bytes: Byte budget per chunk at the MP3 encoding bitrate

        Returns:
            List of audio segment chunks
        """
        bytes_per_ms = (_MP3_BITRATE_KBPS * 1000 / 8) / 1000
        nominal_chunk_ms = min(_MAX_CHUNK_MS, int(max_bytes / bytes_per_ms))

        duration_ms = len(segment)
        if duration_ms <= nominal_chunk_ms:
            return [segment]

        cut_points = [0]
        consumed_until = 0  # end of the silence gap used for the last cut, so its
                             # tail is not rediscovered and picked again next iteration
        cursor = nominal_chunk_ms
        while cursor < duration_ms:
            window_start = max(cursor - _SILENCE_SEARCH_WINDOW_MS, consumed_until)
            window_end = min(cursor + _SILENCE_SEARCH_WINDOW_MS, duration_ms)
            window = segment[window_start:window_end]

            silences = detect_silence(window, min_silence_len=300, silence_thresh=window.dBFS - 16)
            # Only consider silences genuinely within the +/-15s window of the nominal
            # cut point, not just anywhere in the (possibly wider) clamped window.
            candidates = [
                (window_start + s, window_start + e)
                for s, e in silences
                if abs(window_start + (s + e) / 2 - cursor) <= _SILENCE_SEARCH_WINDOW_MS
            ]
            if candidates:
                best_start, best_end = min(candidates, key=lambda c: abs((c[0] + c[1]) / 2 - cursor))
                cut = (best_start + best_end) // 2
                consumed_until = best_end
            else:
                cut = cursor
                consumed_until = cursor

            cut_points.append(cut)
            cursor = cut + nominal_chunk_ms

        cut_points.append(duration_ms)
        return [segment[cut_points[i]:cut_points[i + 1]] for i in range(len(cut_points) - 1)]

    def _transcribe_segment(self, path: str, mode: str, model: str, language: Optional[str]) -> Tuple[str, bool]:
        """Dispatch a single (possibly chunked) audio file to the selected transcription mode"""
        if mode == "chat_audio":
            return self._transcribe_via_chat(path, model, language)
        return self._transcribe_via_stt_endpoint(path, model, language)

    def transcribe_file(
        self,
        file_path: str,
        model: str,
        mode: str = "stt",
        language: Optional[str] = "de",
    ) -> Tuple[str, bool]:
        """
        Transcribe an audio file via OpenRouter

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            mode: "stt" (dedicated transcription endpoint) or "chat_audio"
                  (multimodal chat completions model)
            language: ISO-639-1 language code (e.g. "de"), or None to let
                      the model auto-detect the spoken language

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        temp_paths: List[str] = []
        try:
            prepared_path, fmt = self._prepare_audio(file_path)
            temp_paths.append(prepared_path)

            if os.path.getsize(prepared_path) <= MAX_AUDIO_SIZE:
                return self._transcribe_segment(prepared_path, mode, model, language)

            audio = AudioSegment.from_file(prepared_path)
            chunks = self._split_audio(audio, MAX_AUDIO_SIZE)
            print(f"Audio exceeds {MAX_AUDIO_SIZE} bytes, splitting into {len(chunks)} chunks for transcription.")

            texts = []
            for i, chunk in enumerate(chunks):
                chunk_path = f"{os.path.splitext(prepared_path)[0]}_part{i}.{fmt}"
                temp_paths.append(chunk_path)
                chunk.export(chunk_path, format=fmt, bitrate=_MP3_BITRATE)

                text, success = self._transcribe_segment(chunk_path, mode, model, language)
                if not success:
                    return text, False
                texts.append(text.strip())

            return " ".join(texts), True

        except Exception as e:
            return f"Transcription error: {str(e)}", False

        finally:
            # Delete every temporary artifact (prepared file + chunks), with retries
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    max_retries = 3
                    for i in range(max_retries):
                        try:
                            os.unlink(temp_path)
                            break
                        except PermissionError:
                            if i < max_retries - 1:  # Don't wait on last attempt
                                time.sleep(0.1 * (i + 1))

    def _transcribe_via_stt_endpoint(self, temp_path: str, model: str, language: Optional[str]) -> Tuple[str, bool]:
        """Transcribe using OpenRouter's dedicated /audio/transcriptions endpoint"""
        kwargs = {}
        if language:
            kwargs["language"] = language
        if language == "de":
            kwargs["prompt"] = GERMAN_SPEAKER_PROMPT

        with open(temp_path, 'rb') as f:
            try:
                transcription = self.client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    **kwargs,
                )
                return transcription.text, True
            except NotFoundError:
                print(f"Model {model} not found, using default model 'openai/whisper-1'.")
                f.seek(0)
                transcription = self.client.audio.transcriptions.create(
                    model="openai/whisper-1",
                    file=f,
                    **kwargs,
                )
                return transcription.text, True
            except OpenAIError as e:
                return f"Transcription error: {str(e)}", False

    def _transcribe_via_chat(self, temp_path: str, model: str, language: Optional[str]) -> Tuple[str, bool]:
        """Transcribe by sending the audio as input_audio to a multimodal chat model"""
        audio_format = os.path.splitext(temp_path)[1].lstrip('.').lower()

        with open(temp_path, 'rb') as f:
            base64_audio = base64.b64encode(f.read()).decode("utf-8")

        if language == "de":
            text_prompt = "Transcribe this German audio recording verbatim. Return only the transcript, no commentary."
        elif language:
            text_prompt = f"Transcribe this audio recording (language: {language}) verbatim. Return only the transcript, no commentary."
        else:
            text_prompt = "Transcribe this audio recording verbatim. Return only the transcript, no commentary."

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_prompt,
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": audio_format,
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
