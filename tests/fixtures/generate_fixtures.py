"""
Regenerates the audio fixtures used by tests/test_transcriber.py.

Run with: uv run python tests/fixtures/generate_fixtures.py
"""
import os

from pydub import AudioSegment
from pydub.generators import Sine

FIXTURES_DIR = os.path.dirname(__file__)


def main():
    tone = Sine(440).to_audio_segment(duration=1000).set_channels(1).set_frame_rate(16000)
    tone.export(os.path.join(FIXTURES_DIR, "tone.mp3"), format="mp3", bitrate="64k")

    tone_a = Sine(440).to_audio_segment(duration=8000).set_channels(1).set_frame_rate(16000)
    silence = AudioSegment.silent(duration=2000, frame_rate=16000)
    tone_b = Sine(880).to_audio_segment(duration=8000).set_channels(1).set_frame_rate(16000)
    (tone_a + silence + tone_b).export(os.path.join(FIXTURES_DIR, "tone_silence.wav"), format="wav")


if __name__ == "__main__":
    main()
