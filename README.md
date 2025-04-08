# AI Audio Transcription

A Python-based project for speech recognition, transcription, and text processing with support for multiple languages.

## Project Description

This project enables speech-to-text conversion with the following main features:
- Voice recording via microphone
- Audio file transcription
- Automated text processing with customizable prompt templates
- Support for various AI providers (Groq, OpenAI, OpenRouter)
- User-friendly Streamlit interface
- Robust error handling for API and session-related issues

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
OPENROUTER_API_KEY=your_openrouter_api_key
```
    - You can obtain the API keys here:
      - Groq: https://console.groq.com/keys
      - OpenAI: https://platform.openai.com/api-keys
      - OpenRouter: https://openrouter.ai/keys

4. Install FFmpeg:
    - Windows: Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH
    - macOS: `brew install ffmpeg`
    - Linux: `sudo apt-get install ffmpeg`

## Usage

Start the application with:
```bash
streamlit run src/app.py --server.fileWatcherType=poll
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

### Provider-Specific Notes

#### Groq
- Groq only supports the `whisper-large-v3` model for transcription
- If you select any other transcription model with Groq, it will automatically use `whisper-large-v3`
- The application handles this gracefully and provides appropriate warnings

## How it Works

1. The `AudioTranscriber` class initializes speech recognition
2. Audio is either recorded via microphone or loaded from a file
3. Audio quality is automatically optimized (downsampled to 16kHz mono)
4. Transcription is performed via the chosen AI API (Groq, OpenAI, or OpenRouter)
5. The recognized text is returned and can be further processed
6. The `TextProcessor` class processes the transcribed text based on the chosen prompt template
7. Robust error handling catches and provides user-friendly messages for common issues

## System Requirements

- Python 3.9 or higher
- Working microphone (for live recordings)
- Internet connection (for AI APIs)
- `.env` file with valid API keys
- FFmpeg (in PATH)
- Streamlit 1.44.1 or higher

## Development

### AudioTranscriber Class

The AudioTranscriber class provides functions for speech recognition:

- `transcribe_file(file_path, model)`: Transcribes an audio file using the specified model

### TextProcessor Class

The TextProcessor class processes the transcribed text:

- `process_text(text, prompt_template, model, temperature)`: Processes text based on a prompt template

### Provider Classes

The application uses a provider pattern to support different AI services:

- `GroqAudioProvider`: Handles audio transcription via Groq API
- `GroqTextProvider`: Handles text processing via Groq API
- `OpenAIAudioProvider`: Handles audio transcription via OpenAI API
- `OpenAITextProvider`: Handles text processing via OpenAI API
- `OpenRouterProvider`: Handles text processing via OpenRouter API

### Prompt Templates

Prompt templates are defined in `prompts.py` and can be easily extended or customized.

### Running Tests

The project includes unit tests and integration tests. To run the tests:

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_transcriber.py

# Run with verbose output
python -m pytest -v

# Skip integration tests
python -m pytest -m "not integration"
```

### Project Structure

The project uses a specific import structure:

- **App Code**: Uses relative imports (e.g., `from api_providers.provider_factory import ProviderFactory`)
- **Test Code**: Uses absolute imports with the `src` prefix (e.g., `from src.speech_to_text import AudioTranscriber`)

This structure allows both the app and tests to run correctly. The `conftest.py` file in the tests directory handles the path configuration for tests.

## Troubleshooting

### Common Issues

1. **AppSession Error**:
   - If you encounter an error message like `AttributeError: 'AppSession' object has no attribute '_scriptrunner'`, try restarting the application
   - This is a known issue with certain Streamlit versions and is handled gracefully by the application

2. **Model Compatibility**:
   - Groq only supports `whisper-large-v3` for transcription
   - If you select a different model with Groq, it will automatically use `whisper-large-v3`

3. **API Connection Issues**:
   - If you encounter API connection errors, check your internet connection and API keys
   - Ensure your API keys are correctly set in the `.env` file

## Contributing

Contributions to the project are welcome! Please create a pull request or open an issue for suggestions and bug reports.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
