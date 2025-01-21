from openai import OpenAI, NotFoundError
import os
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
        try:
            audio = AudioSegment.from_file(file_path)
            audio = self.downsample_audio(audio)
            
            temp_path = file_path + '_optimized.wav'
            audio.export(temp_path, format='wav')
            
            with open(temp_path, 'rb') as f:
                try:
                    transcription = self.client.audio.transcriptions.create(
                        model=model,
                        file=f,
                        language="de",
                        response_format="verbose_json",
                        prompt="This is a recording of a German speaker."
                    )
                except NotFoundError:
                    print(f"Model {model} not found, using default model.")
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=f,
                        language="de",
                        response_format="verbose_json",
                        prompt="This is a recording of a German speaker."
                    )
            
            os.unlink(temp_path)
            
            avg_logprob = sum(segment.avg_logprob for segment in transcription.segments) / len(transcription.segments)
            no_speech_prob = sum(segment.no_speech_prob for segment in transcription.segments) / len(transcription.segments)
            
            if avg_logprob < -0.5:
                print("Warning: Low average log probability. Possible transcription issues.")
            if no_speech_prob > 0.5:
                print("Warning: High probability of no speech detected. Possible silence or noise in audio.")
            
            return transcription.text, True
            
        except Exception as e:
            return f"Transcription error: {str(e)}", False
