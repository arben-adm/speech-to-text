# AI Audio Transcription

A Python-based project for speech recognition, transcription, and text processing with support for multiple languages.

## Project Description

This project enables speech-to-text conversion with the following main features:
- Voice recording via microphone
- Audio file transcription
- Automated text processing with customizable prompt templates
- Support for various AI providers (Groq, OpenAI)
- User-friendly Streamlit interface

## Installation with uv

1. Install uv if not already installed:
```bash
pip install uv
```

2. Create a new virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # For Unix-based systems
.venv\Scripts\activate.bat  # For Windows
uv pip install -r requirements.txt
```

3. Create a `.env` file and add your API keys:
```plaintext
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```
    - You can obtain the API keys here:
      - Groq: https://console.groq.com/keys
      - OpenAI: https://platform.openai.com/api-keys

4. Install FFmpeg:
    - Windows: Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH
    - macOS: `brew install ffmpeg`
    - Linux: `sudo apt-get install ffmpeg`

## Usage

Start the application with:
```bash
streamlit run app.py
```

The application offers the following features:

1. **Audio File Upload**:
    - Upload an audio file
    - The file will be transcribed and the text displayed

2. **Microphone Recording**:
    - Record speech directly via microphone
    - The recording will be transcribed and the text displayed

3. **Text Processing**:
    - Select a prompt template for automatic text processing
    - Edit the system prompt if needed
    - The processed text will be displayed and can be downloaded

## How it Works

1. The `AudioTranscriber` class initializes speech recognition
2. Audio is either recorded via microphone or loaded from a file
3. Audio quality is automatically optimized
4. Transcription is performed via the chosen AI API (Groq or OpenAI)
5. The recognized text is returned and can be further processed
6. The `TextProcessor` class processes the transcribed text based on the chosen prompt template

## System Requirements

- Python 3.9 or higher
- Working microphone (for live recordings)
- Internet connection (for AI APIs)
- `.env` file with valid API keys
- FFmpeg (in PATH)

## Development

### AudioTranscriber Class

The AudioTranscriber class provides functions for speech recognition:

- `transcribe_file(file_path)`: Transcribes an audio file
- `transcribe_microphone(timeout=5)`: Transcribes microphone input

### TextProcessor Class

The TextProcessor class processes the transcribed text:

- `process_text(text, prompt_template)`: Processes text based on a prompt template

### Prompt Templates

Prompt templates are defined in `prompts.py` and can be easily extended or customized.

## Contributing

Contributions to the project are welcome! Please create a pull request or open an issue for suggestions and bug reports.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
