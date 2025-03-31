from typing import Tuple, Optional, List, Dict, Any
import os
import time
import json
from pydub import AudioSegment
from openai import OpenAI, OpenAIError, NotFoundError

from .base_provider import BaseAudioProvider, BaseTextProvider
from prompts import PromptTemplate

class GroqAudioProvider(BaseAudioProvider):
    """Groq implementation of the audio provider"""
    
    def __init__(self, api_key: str):
        """
        Initialize the Groq audio provider
        
        Args:
            api_key: Groq API key
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    
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
        Transcribe an audio file using Groq's API
        
        Args:
            file_path: Path to the audio file
            model: Model to use for transcription
            
        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        temp_path = ""
        try:
            # Handle models with provider prefix (e.g., "groq/whisper-large-v3")
            if '/' in model:
                provider, base_model = model.split('/', 1)
                if provider.lower() == 'groq':
                    model = base_model
                else:
                    return f"Error: Model '{model}' is not a Groq model. Please select a Groq model.", False
            
            # Validate model - ensure it's a transcription model
            if 'whisper' not in model.lower():
                return f"Error: '{model}' is not a transcription model. Groq only supports Whisper models for transcription.", False
            
            # Groq currently only supports whisper-large-v3 for transcription
            if model != "whisper-large-v3":
                print(f"Warning: Groq only supports whisper-large-v3 for transcription. Using whisper-large-v3 instead of {model}.")
                model = "whisper-large-v3"
            
            audio = AudioSegment.from_file(file_path)
            audio = self.downsample_audio(audio)
            
            temp_path = file_path + '_optimized.wav'
            audio.export(temp_path, format='wav')
            
            with open(temp_path, 'rb') as f:
                try:
                    # Use Groq's transcription API
                    transcription = self.client.audio.transcriptions.create(
                        model=model,  # Groq only supports whisper-large-v3
                        file=f,
                        language="de",
                        response_format="verbose_json",
                        prompt="This is a recording of a German speaker."
                    )
                    
                    # Process transcription results
                    avg_logprob = sum(segment.avg_logprob for segment in transcription.segments) / len(transcription.segments)
                    no_speech_prob = sum(segment.no_speech_prob for segment in transcription.segments) / len(transcription.segments)
                    
                    if avg_logprob < -0.5:
                        print("Warning: Low average log probability. Possible transcription issues.")
                    if no_speech_prob > 0.5:
                        print("Warning: High probability of no speech detected. Possible silence or noise in audio.")
                    
                    return transcription.text, True
                    
                except OpenAIError as e:
                    error_message = f"Error during transcription: {e.type}"
                    print(error_message)
                    
                    if e.type == "not_found":
                        return f"Error: Model '{model}' not found. Groq only supports whisper-large-v3 for transcription.", False
                    elif e.type == "invalid_request_error":
                        # Get more details from the error
                        error_details = str(e)
                        if "file too large" in error_details.lower():
                            return "Error: Audio file is too large. Please use a shorter audio file.", False
                        else:
                            return f"Error: Invalid request - {error_details}", False
                    elif e.type == "api_connection_error":
                        return "Error: Connection to Groq API failed. Please check your internet connection.", False
                    else:
                        return f"Error: An unknown error occurred - {str(e)}", False
                except Exception as e:
                    print(f"Error with model {model}: {str(e)}")
                    return f"Transcription error: {str(e)}", False
                    
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
        Get available transcription models for Groq
        
        Returns:
            List of available model names
        """
        try:
            # Groq doesn't have a specific endpoint for audio models, so we filter from all models
            response = self.client.models.list()
            
            # Filter for whisper models
            whisper_models = [model.id for model in response.data if 'whisper' in model.id.lower()]
            
            if not whisper_models:
                # Fallback to known models if API doesn't return any
                return [
                    "whisper-large-v3",
                    "whisper-large-v3-turbo",
                    "distil-whisper-large-v3-en"
                ]
            
            return whisper_models
            
        except Exception as e:
            print(f"Error fetching Groq transcription models: {str(e)}")
            # Fallback to known models if API call fails
            return [
                "whisper-large-v3",
                "whisper-large-v3-turbo",
                "distil-whisper-large-v3-en"
            ]


class GroqTextProvider(BaseTextProvider):
    """Groq implementation of the text provider"""
    
    def __init__(self, api_key: str):
        """
        Initialize the Groq text provider
        
        Args:
            api_key: Groq API key
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    
    def process_text(self, text: str, prompt_template: PromptTemplate, model: str = None, temperature: float = 0.2) -> Optional[str]:
        """
        Process text using Groq's API
        
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
            model_name = model if model else "llama-3.3-70b-versatile"
            
            # Handle models with provider prefix (e.g., "groq/llama-3.3-70b-versatile")
            if '/' in model_name:
                provider, base_model = model_name.split('/', 1)
                if provider.lower() == 'groq':
                    model_name = base_model
                else:
                    return f"Error: Model '{model_name}' is not a Groq model. Please select a Groq model."
            
            # Validate model - ensure it's not a transcription model
            if 'whisper' in model_name.lower():
                return f"Error: '{model_name}' is a transcription model, not a chat model. Please select a chat model."
            
            # Get available models to validate
            try:
                available_models = self.get_available_chat_models()
                if model_name not in available_models:
                    # Try to find a similar model as fallback
                    fallback_model = next((m for m in available_models if model_name.lower() in m.lower()), None)
                    if fallback_model:
                        print(f"Model '{model_name}' not found, using '{fallback_model}' instead")
                        model_name = fallback_model
                    else:
                        return f"Error: Model '{model_name}' not found. Available models: {', '.join(available_models[:3])}..."
            except Exception as e:
                # Continue with the provided model if we can't validate
                print(f"Warning: Could not validate model: {str(e)}")
            
            # Process the text
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
            error_message = f"Error during text processing: {e.type}"
            print(error_message)
            
            if e.type == "not_found":
                return f"Error: Model '{model_name}' not found. Please select a different model."
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
                return "Error: Connection to Groq API failed. Please check your internet connection."
            else:
                return f"Error: An unknown error occurred - {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_available_chat_models(self) -> List[str]:
        """
        Get available chat models for Groq
        
        Returns:
            List of available model names
        """
        try:
            # Fetch all models from Groq API
            response = self.client.models.list()
            
            # Filter for chat models (exclude whisper models)
            chat_models = [model.id for model in response.data if 'whisper' not in model.id.lower()]
            
            if not chat_models:
                # Fallback to known models if API doesn't return any
                return [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama-guard-3-8b",
                    "llama3-70b-8192",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it"
                ]
            
            return chat_models
            
        except Exception as e:
            print(f"Error fetching Groq chat models: {str(e)}")
            # Fallback to known models if API call fails
            return [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
