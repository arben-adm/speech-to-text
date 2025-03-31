from typing import Tuple, Optional, List, Dict, Any
import os
import time
import json
from pydub import AudioSegment
from openai import OpenAI, OpenAIError, NotFoundError
import re

from .base_provider import BaseAudioProvider, BaseTextProvider
from prompts import PromptTemplate
from .openai_provider import OpenAIAudioProvider
from .groq_provider import GroqAudioProvider

class OpenRouterAudioProvider(BaseAudioProvider):
    """OpenRouter implementation of the audio provider that routes to OpenAI and Groq"""
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter audio provider
        
        Args:
            api_key: OpenRouter API key
        """
        self.openrouter_api_key = api_key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize the OpenRouter client for text processing
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Initialize provider-specific clients if API keys are available
        self.providers: Dict[str, BaseAudioProvider] = {}
        
        if self.openai_api_key:
            self.providers['openai'] = OpenAIAudioProvider(self.openai_api_key)
        
        if self.groq_api_key:
            self.providers['groq'] = GroqAudioProvider(self.groq_api_key)
    
    def downsample_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Downsample audio to 16kHz mono (required by Whisper)
        
        Args:
            audio_segment: Audio segment to downsample
            
        Returns:
            Downsampled audio segment
        """
        return audio_segment.set_frame_rate(16000).set_channels(1)
    
    def _get_provider_from_model(self, model: str) -> str:
        """
        Extract provider from model string
        
        Args:
            model: Model string (e.g., 'openai/whisper-1', 'groq/whisper-large-v3')
            
        Returns:
            Provider name
        """
        if '/' in model:
            return model.split('/')[0].lower()
        
        # Default mappings for models without explicit provider
        if model.startswith('whisper-large-v3') or model.startswith('llama'):
            return 'groq'
        else:
            return 'openai'  # Default to OpenAI for other models
    
    def _get_base_model_name(self, model: str) -> str:
        """
        Extract base model name from model string
        
        Args:
            model: Model string (e.g., 'openai/whisper-1', 'groq/whisper-large-v3')
            
        Returns:
            Base model name
        """
        if '/' in model:
            return model.split('/', 1)[1]
        return model
    
    def transcribe_file(self, file_path: str, model: str) -> Tuple[str, bool]:
        """
        Transcribe an audio file by routing to the appropriate provider
        
        Args:
            file_path: Path to the audio file
            model: Model to use for transcription (format: 'provider/model' or just 'model')
            
        Returns:
            Tuple containing (transcription_text, success_flag)
        """
        # Extract provider and base model name
        provider_name = self._get_provider_from_model(model)
        base_model = self._get_base_model_name(model)
        
        # Check if we have the provider available
        if provider_name not in self.providers:
            missing_key = f"{provider_name.upper()}_API_KEY"
            return f"Error: {missing_key} not set. Please add it to your .env file.", False
        
        # Route to the appropriate provider
        provider = self.providers[provider_name]
        return provider.transcribe_file(file_path, base_model)
    
    def get_available_transcription_models(self) -> List[str]:
        """
        Get available transcription models from all configured providers
        
        Returns:
            List of available model names with provider prefixes
        """
        models = []
        
        # Add OpenAI models if available
        if 'openai' in self.providers:
            openai_models = ['openai/' + model for model in self.providers['openai'].get_available_transcription_models()]
            models.extend(openai_models)
        
        # Add Groq models if available
        if 'groq' in self.providers:
            groq_models = ['groq/' + model for model in self.providers['groq'].get_available_transcription_models()]
            models.extend(groq_models)
        
        # If no providers are configured, return a message as the first option
        if not models:
            models = ["Please configure OPENAI_API_KEY and/or GROQ_API_KEY in your .env file"]
        
        return models


class OpenRouterTextProvider(BaseTextProvider):
    """OpenRouter implementation of the text provider"""
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter text provider
        
        Args:
            api_key: OpenRouter API key
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
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
            model_name = model if model else "openai/gpt-4o"
            
            # Validate model - ensure it's not a transcription model
            if 'whisper' in model_name.lower():
                return f"Error: '{model_name}' is a transcription model, not a chat model. Please select a chat model."
            
            # Get available models to validate
            try:
                available_models = self.get_available_chat_models()
                if model_name not in available_models and model_name != "openrouter/auto":
                    # Try to find a similar model as fallback
                    provider_prefix = model_name.split('/')[0] if '/' in model_name else None
                    
                    if provider_prefix:
                        # Look for models from the same provider
                        provider_models = [m for m in available_models if m.startswith(f"{provider_prefix}/")]
                        if provider_models:
                            fallback_model = provider_models[0]
                            print(f"Model '{model_name}' not found, using '{fallback_model}' instead")
                            model_name = fallback_model
                        else:
                            # If no models from that provider, use auto routing
                            print(f"No models found from provider '{provider_prefix}', using auto routing")
                            model_name = "openrouter/auto"
                    else:
                        # If no provider prefix, use auto routing
                        print(f"Model '{model_name}' not found, using auto routing")
                        model_name = "openrouter/auto"
            except Exception as e:
                # Continue with the provided model if we can't validate
                print(f"Warning: Could not validate model: {str(e)}")
            
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
            # Fetch all models from OpenRouter API
            response = self.client.models.list()
            
            # Extract model IDs
            models = [model.id for model in response.data]
            
            # Add special routing option
            models.append("openrouter/auto")
            
            if not models or len(models) <= 1:  # Only the auto option
                # Fallback to known models if API doesn't return any
                return [
                    # OpenAI models
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "openai/o1",
                    "openai/o1-mini",
                    "openai/gpt-4-turbo",
                    # Groq models
                    "groq/llama-3.3-70b-versatile",
                    "groq/llama-3.1-8b-instant",
                    # Anthropic models
                    "anthropic/claude-3-opus-20240229",
                    "anthropic/claude-3-sonnet-20240229",
                    # Special routing
                    "openrouter/auto"
                ]
            
            return models
            
        except Exception as e:
            print(f"Error fetching OpenRouter models: {str(e)}")
            # Fallback to known models if API call fails
            return [
                # OpenAI models
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/o1",
                "openai/o1-mini",
                "openai/gpt-4-turbo",
                # Groq models
                "groq/llama-3.3-70b-versatile",
                "groq/llama-3.1-8b-instant",
                # Anthropic models
                "anthropic/claude-3-opus-20240229",
                "anthropic/claude-3-sonnet-20240229",
                # Special routing
                "openrouter/auto"
            ]
