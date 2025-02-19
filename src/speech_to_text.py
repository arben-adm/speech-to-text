from openai import OpenAI, NotFoundError
import os
import time
from pydub import AudioSegment
from typing import Tuple

class AudioTranscriber:
    def __init__(self, provider: str, api_key: str):
        """Initializes AudioTranscriber with chosen provider"""
        self.provider = provider.lower()
        base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def downsample_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """Downsamples audio to 16kHz mono"""
        return audio_segment.set_frame_rate(16000).set_channels(1)
    
    def transcribe_file(self, file_path: str, model: str) -> Tuple[str, bool]:
        """Transcribes audio using selected provider"""
        temp_path = ""
        try:
            audio = AudioSegment.from_file(file_path)
            audio = self.downsample_audio(audio)
            
            temp_path = file_path + '_optimized.wav'
            audio.export(temp_path, format='wav')
            
            with open(temp_path, 'rb') as f:
                try:
                    if self.provider == 'groq':
                        # Groq specific transcription
                        transcription = self.client.audio.transcriptions.create(
                            model="whisper-large-v3",  # Groq only supports this model
                            file=f,
                            language="de",
                            response_format="verbose_json",
                            prompt="This is a recording of a German speaker."
                        )
                    else:
                        # OpenAI transcription
                        transcription = self.client.audio.transcriptions.create(
                            model=model,
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
                    
                except NotFoundError:
                    print(f"Model {model} not found, using default model.")
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-large-v3",
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
