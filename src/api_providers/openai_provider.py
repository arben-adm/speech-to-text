from typing import Tuple, Optional, List, Dict, Any
import os
import time
import json
from pydub import AudioSegment
from openai import OpenAI, OpenAIError, NotFoundError
from .base_provider import BaseAudioProvider, BaseTextProvider
from prompts import PromptTemplate

class OpenAIAudioProvider(BaseAudioProvider):
    """OpenAI implementation of the audio provider"""

    def __init__(self, api_key: str):
        """
        Initialize the OpenAI audio provider

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)

    def downsample_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Downsample audio to 16kHz mono (required by Whisper)

        Args:
            audio_segment: Audio segment to downsample

        Returns:
            Downsampled audio segment
        """
        return audio_segment.set_frame_rate(16000).set_channels(1)

    def transcribe_file(self, file_path: str, model: str) -> Tuple[str, bool]:
        """
        Transcribe an audio file using OpenAI's API

        Args:
            file_path: Path to the audio file
            model: Model to use for transcription

        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        temp_path = ""
        try:
            audio = AudioSegment.from_file(file_path)
            audio = self.downsample_audio(audio)

            temp_path = file_path + '_optimized.wav'
            audio.export(temp_path, format='wav')

            with open(temp_path, 'rb') as f:
                try:
                    # Determine the appropriate response format based on the model
                    # New GPT-4o transcription models only support json or text
                    if 'gpt-4o' in model.lower() and 'transcribe' in model.lower():
                        response_format = "json"
                        transcription = self.client.audio.transcriptions.create(
                            model=model,
                            file=f,
                            language="de",
                            response_format=response_format,
                            prompt="This is a recording of a German speaker."
                        )

                        # For json format, the text is directly accessible
                        return transcription.text, True
                    else:
                        # For whisper models, use verbose_json for additional metadata
                        response_format = "verbose_json"
                        transcription = self.client.audio.transcriptions.create(
                            model=model,
                            file=f,
                            language="de",
                            response_format=response_format,
                            prompt="This is a recording of a German speaker."
                        )

                        # Process transcription results for whisper models
                        avg_logprob = sum(segment.avg_logprob for segment in transcription.segments) / len(transcription.segments)
                        no_speech_prob = sum(segment.no_speech_prob for segment in transcription.segments) / len(transcription.segments)

                        if avg_logprob < -0.5:
                            print("Warning: Low average log probability. Possible transcription issues.")
                        if no_speech_prob > 0.5:
                            print("Warning: High probability of no speech detected. Possible silence or noise in audio.")

                        return transcription.text, True

                except NotFoundError:
                    print(f"Model {model} not found, using default model.")
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",  # Default OpenAI model
                        file=f,
                        language="de",
                        response_format="verbose_json",
                        prompt="This is a recording of a German speaker."
                    )
                    return transcription.text, True

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

    def get_available_transcription_models(self) -> List[str]:
        """
        Get available transcription models for OpenAI

        Returns:
            List of available model names
        """
        try:
            # Fetch all models from OpenAI API
            response = self.client.models.list()

            # Filter for audio/transcription models
            audio_models = []

            # Check for whisper models
            whisper_models = [model.id for model in response.data if 'whisper' in model.id.lower()]
            audio_models.extend(whisper_models)

            # Check for new GPT-4o transcription models
            gpt4o_transcribe_models = [model.id for model in response.data
                                      if ('gpt-4o' in model.id.lower() and 'transcribe' in model.id.lower())]
            audio_models.extend(gpt4o_transcribe_models)

            # If no models found from API, use known models
            if not audio_models:
                # Include both whisper-1 and the new GPT-4o transcription models
                return [
                    "whisper-1",
                    "gpt-4o-mini-transcribe",
                    "gpt-4o-transcribe"
                ]

            # Make sure the new models are included even if not returned by the API
            if "gpt-4o-mini-transcribe" not in audio_models:
                audio_models.append("gpt-4o-mini-transcribe")

            if "gpt-4o-transcribe" not in audio_models:
                audio_models.append("gpt-4o-transcribe")

            return audio_models

        except Exception as e:
            print(f"Error fetching OpenAI transcription models: {str(e)}")
            # Fallback to known models if API call fails
            return [
                "whisper-1",
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe"
            ]


class OpenAITextProvider(BaseTextProvider):
    """OpenAI implementation of the text provider"""

    def __init__(self, api_key: str):
        """
        Initialize the OpenAI text provider

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)

    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """
        Process text using OpenAI's API

        Args:
            text: Text to process
            prompt_template: Prompt template to use
            model: Model to use for processing (optional)
            temperature: Temperature parameter for generation (optional)

        Returns:
            Processed text or None if processing failed
        """
        try:
            model_name = model if model else "gpt-4o-mini"

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
                temperature=temperature
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"Error during text processing: {e.type}")
            if e.type == "not_found":
                return "Error: Model not found. Please check the model name."
            elif e.type == "invalid_request_error":
                return "Error: Invalid request. Please check the parameters."
            elif e.type == "api_connection_error":
                return "Error: Connection to API server failed. Please check your internet connection."
            else:
                return "Error: An unknown error occurred."

    def get_available_chat_models(self) -> List[str]:
        """
        Get available chat models for OpenAI

        Returns:
            List of available model names
        """
        try:
            # Fetch all models from OpenAI API
            response = self.client.models.list()

            # Filter for chat models (exclude whisper models)
            chat_models = [model.id for model in response.data
                          if ('gpt' in model.id.lower() or
                              'o1' in model.id.lower() or
                              'o3' in model.id.lower()) and
                             'whisper' not in model.id.lower()]

            if not chat_models:
                # Fallback to known models if API doesn't return any
                return [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "o1",
                    "o1-mini",
                    "o3-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo"
                ]

            return chat_models

        except Exception as e:
            print(f"Error fetching OpenAI chat models: {str(e)}")
            # Fallback to known models if API call fails
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "o1",
                "o1-mini",
                "o3-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
