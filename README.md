# AI Audio Transcription

A Python-based project for speech recognition, transcription, and text processing with support for multiple languages.

## Project Description

This project enables speech-to-text conversion with the following main features:
- Voice recording via microphone
- Audio file transcription
- Automated text processing with customizable prompt templates
- Runs entirely on [OpenRouter](https://openrouter.ai) — one API key for both transcription and chat, across hundreds of models
- Toggleable transcription mode: OpenRouter's dedicated Speech-to-Text endpoint, or a multimodal chat completions model fed the raw audio
- Toggleable automatic text processing after transcription
- AI Agent with MCP (Model Context Protocol) server integration
- Chat interface for direct interaction with the AI Agent
- User-friendly Streamlit interface
- Robust error handling for API and session-related issues

## Installation with uv

1. Install uv if not already installed:
```bash
pip install uv
```

2. Sync the virtual environment and install all dependencies (declared in `pyproject.toml`, pinned in `uv.lock`):
```bash
uv sync
```
This creates `.venv` and installs the exact locked versions, including dev dependencies (`pytest`, `pytest-asyncio`). No manual `venv`/`activate`/`pip install` steps needed — prefix commands with `uv run` (e.g. `uv run streamlit run src/app.py`) to run them inside the synced environment, or activate `.venv` yourself as usual.

3. Create a `.env` file and add your API key:
```plaintext
OPENROUTER_API_KEY=your_openrouter_api_key
```
    - Get your key here: https://openrouter.ai/keys
    - This is the only API key the app needs. All transcription and chat requests are routed through OpenRouter (`https://openrouter.ai/api/v1`), which in turn gives you access to OpenAI, Groq-hosted, Anthropic, Google, and many other models via a single key.

4. Install FFmpeg:
    - Windows: Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH
    - macOS: `brew install ffmpeg`
    - Linux: `sudo apt-get install ffmpeg`

## Usage

Start the application with:
```bash
uv run streamlit run src/app.py --server.fileWatcherType=poll
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

4. **MCP Configuration**:
    - Configure Model Context Protocol (MCP) servers
    - Add and remove MCP servers with custom configurations

5. **MCP Tools**:
    - Explore and use available tools from connected MCP servers
    - Execute MCP tools with custom parameters

6. **AI Agent**:
    - Chat directly with an AI agent that has access to both speech-to-text functionality and MCP tools
    - Process audio files and recordings with the agent
    - Customize the agent's system prompt and view available tools

### Interactive Toggles

The transcription tab (and the agent's "Process audio" tab) exposes two toggles:

- **Transcription mode**:
  - *Speech-to-Text endpoint* — uses OpenRouter's dedicated `/audio/transcriptions` endpoint (Whisper-class and token-priced STT models). Fast, cheap, produces a verbatim transcript.
  - *Chat Completions (multimodal audio)* — sends the audio as `input_audio` content to a multimodal chat model (e.g. Gemini, GPT-4o Audio). Useful when you want the model to reason about the audio rather than just transcribe it verbatim.
  - The available model list updates automatically based on the selected mode.
- **Automatically process transcription** — when enabled (default), the transcript is immediately run through the selected prompt template. Disable it to only get the raw transcript (with a download button).

## How it Works

1. The `AudioTranscriber` class initializes speech recognition
2. Audio is either recorded via microphone or loaded from a file
3. Audio quality is automatically optimized (downsampled to 16kHz mono)
4. Transcription is performed via OpenRouter, using either its dedicated transcription endpoint or a multimodal chat model, depending on the selected mode
5. The recognized text is returned and, if auto-processing is enabled, further processed
6. The `TextProcessor` class processes the transcribed text based on the chosen prompt template (also via OpenRouter)
7. The `MCPClient` class connects to configured MCP servers and provides access to their tools
8. The `Agent` and `SpeechAgent` classes combine the speech-to-text functionality with MCP server tools
9. Robust error handling catches and provides user-friendly messages for common issues

## System Requirements

- Python 3.10 or higher (required by the `mcp` package)
- Working microphone (for live recordings)
- Internet connection (for the OpenRouter API)
- `.env` file with a valid `OPENROUTER_API_KEY`
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

The application uses a provider pattern, with OpenRouter as the sole backend:

- `OpenRouterAudioProvider`: Handles audio transcription via OpenRouter, either through the dedicated `/audio/transcriptions` endpoint ("stt" mode) or by sending audio to a multimodal chat model ("chat_audio" mode)
- `OpenRouterTextProvider`: Handles text processing via OpenRouter's chat completions API

Only one API key (`OPENROUTER_API_KEY`) is required; OpenRouter routes to whichever underlying model the selected model slug (e.g. `openai/whisper-1`, `anthropic/claude-sonnet-4.5`) points to.

### MCP Integration

The MCP (Model Context Protocol) integration enables connecting to external tools:

- `MCPClient`: Manages connections to MCP servers and tool execution
- `MCPServerIntegration`: Provides easy configuration and management of MCP servers

### Agent Classes

The application includes AI agent capabilities:

- `Agent`: Base class for AI agents that can use both local tools and MCP server tools
- `SpeechAgent`: Extension of Agent that integrates speech-to-text functionality
- `Tool`: Base class for implementing local tools that agents can use

### Prompt Templates

Prompt templates are defined in `prompts.py` and can be easily extended or customized.

### Running Tests

The project includes unit tests and integration tests. To run the tests:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_transcriber.py

# Run with verbose output
uv run pytest -v

# Skip integration tests
uv run pytest -m "not integration"
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
   - If a selected model isn't found, the app falls back to `openai/whisper-1` for transcription
   - Not every model supports both transcription modes — switching the "Transcription mode" toggle refreshes the model list to only show models that support the selected mode

3. **API Connection Issues**:
   - If you encounter API connection errors, check your internet connection and API key
   - Ensure `OPENROUTER_API_KEY` is correctly set in the `.env` file

## Contributing

Contributions to the project are welcome! Please create a pull request or open an issue for suggestions and bug reports.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
