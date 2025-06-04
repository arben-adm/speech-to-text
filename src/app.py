import streamlit as st
import asyncio
from speech_to_text import AudioTranscriber
from text_processors import TextProcessor
from prompts import AVAILABLE_PROMPTS, PromptTemplate
from streamlit_mic_recorder import mic_recorder
import uuid
import tempfile
from dotenv import load_dotenv
import os
import time
import json
from mcp_client import get_mcp_client, run_async
from agents.agent import Agent
from agents.speech_agent import SpeechAgent
from agents.tools.think import ThinkTool

load_dotenv()

class TranscriptionApp:
    def __init__(self):
        if 'provider' not in st.session_state:
            st.session_state.provider = 'groq'

        # Cache für transkribierte Texte
        if 'transcription_cache' not in st.session_state:
            st.session_state.transcription_cache = {}

        # Cache für verarbeitete Texte
        if 'processed_text_cache' not in st.session_state:
            st.session_state.processed_text_cache = {}
            
        # Chat-Nachrichten für den Agenten
        if 'agent_messages' not in st.session_state:
            st.session_state.agent_messages = []
            
        # MCP-Client initialisieren
        self.mcp_client = get_mcp_client()
        
        # MCP-Server initialisieren (wenn noch nicht verbunden)
        if 'mcp_connected' not in st.session_state:
            st.session_state.mcp_connected = False
            try:
                run_async(self.mcp_client.connect_to_servers())
                st.session_state.mcp_connected = True
            except Exception as e:
                print(f"Error connecting to MCP servers: {str(e)}")

        # Lazy Loading für Provider
        self.providers = {}

        # Initialize all providers and cache their models at startup
        self.initialize_all_providers()

        # Setup the selected provider
        self.setup_provider()
        
        # Initialisiere den Agenten
        self.initialize_agent()

    def initialize_all_providers(self):
        """Initialize all providers and cache their models at startup"""
        if 'cached_models' not in st.session_state:
            st.session_state.cached_models = {}

            with st.spinner("Initializing providers and loading models..."):
                # Initialize all providers
                providers = ['groq', 'openai', 'openrouter']

                for provider in providers:
                    api_key = os.getenv(f"{provider.upper()}_API_KEY")
                    if not api_key:
                        st.session_state.cached_models[provider] = {
                            'chat': [f"No {provider.upper()}_API_KEY found in .env file"],
                            'transcription': [f"No {provider.upper()}_API_KEY found in .env file"]
                        }
                        continue

                    try:
                        # Initialize provider
                        transcriber = AudioTranscriber(provider=provider, api_key=api_key)
                        text_processor = TextProcessor(provider=provider, api_key=api_key)

                        # Cache models
                        st.session_state.cached_models[provider] = {
                            'chat': text_processor.get_available_models(),
                            'transcription': transcriber.get_available_models()
                        }
                    except Exception as e:
                        st.error(f"Error initializing {provider}: {str(e)}")
                        st.session_state.cached_models[provider] = {
                            'chat': [f"Error loading {provider} models"],
                            'transcription': [f"Error loading {provider} models"]
                        }

    def setup_provider(self):
        """Initializes the selected provider"""
        provider = st.session_state.provider.lower()
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

        # Ensure environment variables are loaded for OpenRouter to access OpenAI and Groq
        if provider == 'openrouter':
            # Check if API keys are available and show warnings if not
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("OpenAI API key not found. OpenAI models will not be available for transcription via OpenRouter.")

            if not os.getenv("GROQ_API_KEY"):
                st.warning("Groq API key not found. Groq models will not be available for transcription via OpenRouter.")

        self.transcriber = AudioTranscriber(provider=provider, api_key=api_key)
        self.text_processor = TextProcessor(provider=provider, api_key=api_key)
        
    def initialize_agent(self):
        """Initialize the SpeechAgent with current provider settings"""
        if 'agent' not in st.session_state:
            provider = st.session_state.provider.lower()
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            
            # Get default models for the current provider
            models = st.session_state.cached_models.get(provider, {})
            chat_models = models.get('chat', [])
            transcription_models = models.get('transcription', [])
            
            default_chat_model = chat_models[0] if chat_models and chat_models[0].startswith("No") is False else "default-model"
            default_transcription_model = transcription_models[0] if transcription_models and transcription_models[0].startswith("No") is False else "default-model"
            
            try:
                # Create the SpeechAgent with the current provider and models
                st.session_state.agent = SpeechAgent(
                    name="SpeechAssistant",
                    system="You are a helpful assistant that can transcribe and process audio data.",
                    provider=provider,
                    api_key=api_key,
                    transcription_model=default_transcription_model,
                    chat_model=default_chat_model
                )
                
                # Connect to MCP servers
                st.session_state.agent.connect()
                
            except Exception as e:
                print(f"Error initializing the agent: {str(e)}")
                st.session_state.agent = None

    def setup_ui(self):
        st.title("AI Audio Transcription")

        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["Transcription", "MCP Configuration", "MCP Tools", "AI Agent"])

        with tab1:
            self.setup_transcription_ui()

        with tab2:
            self.setup_mcp_config()

        with tab3:
            self.show_mcp_tools()
            
        with tab4:
            self.setup_agent_ui()

    def setup_mcp_config(self):
        """Displays and edits the MCP configuration"""
        st.header("MCP-Server-Konfiguration")
        
        # Load current configuration
        config_path = "mcp_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}
        
        # Show current servers
        if config["mcpServers"]:
            st.write("Configured servers:")
            for server_name, server_config in config["mcpServers"].items():
                with st.expander(f"Server: {server_name}"):
                    st.code(json.dumps(server_config, indent=2))
        else:
            st.info("No MCP servers configured")
        
        # Server form
        with st.expander("Add new server"):
            server_name = st.text_input("Server name")
            command = st.text_input("Command", value="uv")
            
            # Arguments as list
            args_str = st.text_area("Arguments (JSON array)", value='["--directory", "C:/path/to/server", "run", "server.py"]')
            
            # Environment variables as dictionary
            env_str = st.text_area("Environment variables (JSON object)", value='{"API_KEY": "your-api-key"}')
            
            if st.button("Add server"):
                try:
                    args = json.loads(args_str)
                    env = json.loads(env_str)
                    
                    if not server_name:
                        st.error("Please provide a server name")
                    else:
                        config["mcpServers"][server_name] = {
                            "command": command,
                            "args": args,
                            "env": env
                        }
                        
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                        
                        st.success(f"Server {server_name} successfully added")
                        st.info("Please restart the app to use the new servers")
                except json.JSONDecodeError as e:
                    st.error(f"Error in JSON format: {str(e)}")
        
        # Remove server
        if config["mcpServers"]:
            with st.expander("Remove server"):
                server_to_remove = st.selectbox(
                    "Select server",
                    options=list(config["mcpServers"].keys())
                )
                
                if st.button("Remove server") and server_to_remove:
                    del config["mcpServers"][server_to_remove]
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.success(f"Server {server_to_remove} successfully removed")
                    st.info("Please restart the app to apply the changes")

    def show_mcp_tools(self):
        """Displays available MCP tools"""
        st.header("MCP-Tools")
        
        # Check if MCP servers are connected
        if not st.session_state.mcp_connected:
            st.warning("No MCP servers connected. Please configure and start the servers.")
            return
        
        # Get available servers
        servers = run_async(self.mcp_client.list_servers())
        
        if not servers:
            st.info("No MCP servers connected")
            return
        
        # Select server and tool
        col1, col2 = st.columns(2)
        
        with col1:
            selected_server = st.selectbox(
                "Select server",
                options=servers
            )
        
        # Get tools for the selected server
        tools = run_async(self.mcp_client.list_tools(selected_server))
        
        if not tools:
            st.info(f"No tools available for server {selected_server}")
            return
        
        with col2:
            selected_tool = st.selectbox(
                "Select tool",
                options=[tool["name"] for tool in tools],
                format_func=lambda x: next((t["description"] for t in tools if t["name"] == x), x)
            )
        
        # Get the selected tool
        tool_info = next((t for t in tools if t["name"] == selected_tool), None)
        
        if tool_info:
            st.subheader(f"Tool: {tool_info['name']}")
            st.write(f"Description: {tool_info['description']}")
            
            # Form for tool parameters
            st.subheader("Tool Parameters")
            
            # Here we need to get parameters from the schema
            # Since we can't access the schema, we'll use a simple text field
            params_str = st.text_area("Parameters (JSON object)", value="{}")
            
            if st.button("Execute tool"):
                try:
                    params = json.loads(params_str)
                    with st.spinner(f"Executing {selected_tool}..."):
                        result = run_async(self.mcp_client.call_tool(
                            selected_server, 
                            selected_tool, 
                            params
                        ))
                    
                    if "error" in result:
                        st.error(f"Error executing the tool: {result['error']}")
                    else:
                        st.success("Tool executed successfully")
                        
                        # Show the result
                        st.subheader("Result:")
                        if result["isError"]:
                            st.error("Tool execution failed")
                        
                        for content_item in result["content"]:
                            if content_item["type"] == "text":
                                st.write(content_item["text"])
                            # More types could be added here
                
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for parameters")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    def setup_agent_ui(self):
        """UI for the AI Agent with integration of Speech-to-Text functionality"""
        st.header("AI Agent with Speech-to-Text")
        
        # Check if the agent was initialized
        if 'agent' not in st.session_state or st.session_state.agent is None:
            st.warning("The agent could not be initialized. Please check your API keys.")
            return
        
        # Show agent configuration
        with st.expander("Agent Configuration"):
            # Edit system prompt
            system_prompt = st.text_area(
                "System prompt for the agent:",
                value="You are a helpful assistant that can transcribe and process audio data.",
                height=100
            )
            
            # Show available tools
            tools = st.session_state.agent.get_available_tools()
            st.subheader("Available Tools")
            
            # Group tools by type (local vs. MCP)
            local_tools = [t for t_id, t in tools.items() if t["type"] == "local"]
            mcp_tools = [t for t_id, t in tools.items() if t["type"] == "mcp"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Local Tools:")
                for tool in local_tools:
                    st.write(f"- **{tool['name']}**: {tool['description']}")
            
            with col2:
                st.write("MCP Server Tools:")
                if mcp_tools:
                    for tool in mcp_tools:
                        st.write(f"- **{tool['name']}** ({tool['server']}): {tool['description']}")
                else:
                    st.info("No MCP tools available")
            
            # Button to update agent configuration
            if st.button("Update agent configuration"):
                st.session_state.agent.system = system_prompt
                st.success("Agent configuration updated")
        
        # Tabs for different agent interactions
        agent_tab1, agent_tab2 = st.tabs(["Chat with the agent", "Process audio"])
        
        with agent_tab1:
            self.setup_agent_chat_ui()
            
        with agent_tab2:
            self.setup_agent_audio_ui()
    
    def setup_agent_chat_ui(self):
        """Chat interface for the agent"""
        st.subheader("Chat with the agent")
        
        # Chat-Verlauf anzeigen
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.agent_messages:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # User input
        user_input = st.chat_input("Message to the agent...")
        
        if user_input:
            # Benutzer-Nachricht anzeigen
            st.session_state.agent_messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            # Placeholder für die Antwort des Agenten
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                async def stream_response(chunk):
                    # This function is called for each chunk of the stream
                    response_placeholder.markdown(chunk)
                
                # Process the request with the agent
                with st.spinner("Agent is working..."):
                    response = run_async(st.session_state.agent.process(user_input, stream_response))
                
                # Save the response in the chat history
                st.session_state.agent_messages.append({"role": "assistant", "content": response})
    
    def setup_agent_audio_ui(self):
        """Audio processing with the agent"""
        st.subheader("Process audio")
        
        # Model selection
        models = self.get_available_models()
        col1, col2 = st.columns(2)
        with col1:
            chat_model = st.selectbox(
                "Chat model:",
                options=models['chat'],
                key="agent_chat_model"
            )
        with col2:
            transcription_model = st.selectbox(
                "Transcription model:",
                options=models['transcription'],
                key="agent_transcription_model"
            )
        
        # Direct system prompt for the agent
        st.subheader("Agent instructions")
        system_prompt = st.text_area(
            "Give the agent instructions on how to process the audio file:",
            value="Transcribe the audio file and summarize the key points.",
            height=100,
            key="agent_system_prompt"
        )
        
        # Tabs for input methods
        tab1, tab2 = st.tabs(["Upload file", "Microphone recording"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload audio",
                type=['mp3','wav','m4a'],
                key="agent_audio_upload"
            )
            if uploaded_file:
                self.handle_agent_file_upload(uploaded_file, transcription_model, chat_model, system_prompt)
                
        with tab2:
            st.write("Record your voice directly:")
            audio = mic_recorder(
                start_prompt="🎤 Start recording",
                stop_prompt="⏹️ Stop recording",
                just_once=True,
                use_container_width=True,
                key="agent_mic_recorder"
            )
            
            if audio:
                st.audio(audio['bytes'])
                self.handle_agent_recording(audio['bytes'], transcription_model, chat_model, system_prompt)
    
    def handle_agent_file_upload(self, uploaded_file, transcription_model, chat_model, system_prompt):
        """Processes an uploaded audio file with the agent"""
        with st.spinner("Verarbeite Audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                # Datei speichern
                tmp_file.write(uploaded_file.getvalue())
                
                # Callback-Funktion für Fortschrittsaktualisierungen
                results_container = st.container()
                progress = st.progress(0)
                
                async def update_progress(status_type, status_text, additional_info=""):
                    if status_type == "status":
                        if status_text == "Transcribing audio...":
                            progress.progress(25)
                        elif status_text == "Processing text...":
                            progress.progress(75)
                    elif status_type == "transcription":
                        progress.progress(50)
                        with results_container:
                            st.subheader("Transkription:")
                            st.write(status_text)
                    elif status_type == "processed":
                        progress.progress(100)
                        with results_container:
                            st.subheader("Verarbeiteter Text:")
                            st.write(status_text)
                    elif status_type == "error":
                        progress.progress(100)
                        with results_container:
                            st.error(status_text)
                
                # Transkribiere und verarbeite mit dem Agenten
                result = run_async(st.session_state.agent.transcribe_and_process(
                    audio_bytes=uploaded_file.getvalue(),
                    transcription_model=transcription_model,
                    chat_model=chat_model,
                    system_prompt=system_prompt,
                    callback=update_progress
                ))
                
                # Zeige Download-Button an, wenn erfolgreich
                if "processed_text" in result:
                    with results_container:
                        st.download_button(
                            label="Verarbeiteten Text herunterladen",
                            data=result["processed_text"],
                            file_name="processed_text.txt",
                            mime="text/plain"
                        )
                
                # Aufräumen
                os.unlink(tmp_file.name)
    
    def handle_agent_recording(self, audio_bytes, transcription_model, chat_model, system_prompt):
        """Verarbeitet eine Mikrofon-Aufnahme mit dem Agenten"""
        with st.spinner("Verarbeite Audio..."):
            # Callback-Funktion für Fortschrittsaktualisierungen
            results_container = st.container()
            progress = st.progress(0)
            
            async def update_progress(status_type, status_text, additional_info=""):
                if status_type == "status":
                    if status_text == "Transcribing audio...":
                        progress.progress(25)
                    elif status_text == "Processing text...":
                        progress.progress(75)
                elif status_type == "transcription":
                    progress.progress(50)
                    with results_container:
                        st.subheader("Transkription:")
                        st.write(status_text)
                elif status_type == "processed":
                    progress.progress(100)
                    with results_container:
                        st.subheader("Verarbeiteter Text:")
                        st.write(status_text)
                elif status_type == "error":
                    progress.progress(100)
                    with results_container:
                        st.error(status_text)
            
            # Transkribiere und verarbeite mit dem Agenten
            result = run_async(st.session_state.agent.transcribe_and_process(
                audio_bytes=audio_bytes,
                transcription_model=transcription_model,
                chat_model=chat_model,
                system_prompt=system_prompt,
                callback=update_progress
            ))
            
            # Zeige Download-Button an, wenn erfolgreich
            if "processed_text" in result:
                with results_container:
                    st.download_button(
                        label="Verarbeiteten Text herunterladen",
                        data=result["processed_text"],
                        file_name="processed_text.txt",
                        mime="text/plain"
                    )

    def setup_transcription_ui(self):
        """Original UI für Transkription"""
        # Provider selection
        col1, col2 = st.columns([2,1])
        with col1:
            provider = st.selectbox(
                "Select AI Provider:",
                options=['Groq', 'OpenAI', 'OpenRouter'],
                index=0 if st.session_state.provider == 'groq' else
                      1 if st.session_state.provider == 'openai' else 2,
                help="OpenRouter benötigt OpenAI und/oder Groq API-Schlüssel für die Transkription"
            )

            if provider.lower() != st.session_state.provider:
                st.session_state.provider = provider.lower()
                self.setup_provider()
                
                # Aktualisiere auch den Agenten bei Providerwechsel
                if 'agent' in st.session_state:
                    # Trenne bestehende Verbindungen
                    st.session_state.agent.disconnect()
                    
                    # Entferne den Agenten aus dem Session State
                    del st.session_state.agent
                    
                    # Initialisiere den Agenten neu
                    self.initialize_agent()
                
                st.rerun()

        with col2:
            st.markdown(f"**Active Provider:** {provider}")

        # Show OpenRouter help message
        if provider.lower() == 'openrouter':
            st.info("OpenRouter verwendet OpenAI und Groq für die Transkription. Bitte stellen Sie sicher, dass die entsprechenden API-Schlüssel in Ihrer .env-Datei konfiguriert sind.")

        # Update model selection
        models = self.get_available_models()
        col1, col2 = st.columns(2)
        with col1:
            chat_model = st.selectbox(
                "Select Chat Model:",
                options=models['chat']
            )
        with col2:
            transcription_model = st.selectbox(
                "Select Transcription Model:",
                options=models['transcription']
            )

        # Prompt Template selection before tabs
        st.subheader("Automatic Text Processing")
        col1, col2 = st.columns([3, 1])
        with col1:
            prompt = st.selectbox(
                "Choose a processing option:",
                options=AVAILABLE_PROMPTS,
                format_func=lambda x: x.name,
                help="Select how the transcribed text should be processed"
            )
        with col2:
            st.markdown(f"**Description:** {prompt.description}")

        # Display and edit System Prompt
        with st.expander("Show/Edit System Prompt"):
            edited_system_prompt = st.text_area("System Prompt", value=prompt.system_prompt, height=300)
            if edited_system_prompt != prompt.system_prompt:
                prompt = PromptTemplate(name=prompt.name, description=prompt.description, system_prompt=edited_system_prompt)

        # Tabs for input methods
        tab1, tab2, tab3 = st.tabs(["File Upload", "Microphone Recording", "Text Input"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Audio",
                type=['mp3','wav','m4a']
            )
            if uploaded_file:
                self.handle_file_upload(uploaded_file, transcription_model, chat_model, prompt)

        with tab2:
            st.write("Record your voice directly:")
            audio = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=True,
                use_container_width=True
            )

            if audio:
                st.audio(audio['bytes'])
                self.handle_recording(audio['bytes'], transcription_model, chat_model, prompt)

        with tab3:
            st.write("Enter your text directly:")

            text_input = st.chat_input("Type your message here...")

            if text_input:
                is_valid, message = self.validate_text_input(text_input)
                if not is_valid:
                    st.error(message)
                else:
                    token_count = self.count_tokens(text_input)
                    st.info(f"Approximate tokens: {token_count}")
                    try:
                        with st.spinner("Processing Text..."):
                            processed_text = self.text_processor.process_text(
                                text_input,
                                prompt,
                                model=chat_model
                            )
                        if processed_text:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original Text:")
                                st.write(text_input)
                            with col2:
                                st.subheader(f"Processed Text ({prompt.name}):")
                                st.write(processed_text)
                            st.download_button(
                                label="Download Processed Text",
                                data=processed_text,
                                file_name="processed_text.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error processing text: {str(e)}")
                        if "rate limits exceeded" in str(e).lower():
                            st.warning("Please wait a moment before submitting another request.")

    def get_available_models(self):
        """Returns available models based on the provider"""
        provider = st.session_state.provider.lower()

        # Use cached models if available
        if 'cached_models' in st.session_state and provider in st.session_state.cached_models:
            return st.session_state.cached_models[provider]

        # Fallback to direct API calls if cache is not available
        return {
            'chat': self.text_processor.get_available_models(),
            'transcription': self.transcriber.get_available_models()
        }

    def handle_file_upload(self, uploaded_file, model, chat_model, prompt):
        with st.spinner("Processing Audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())

                text, success = self.transcriber.transcribe_file(
                    tmp_file.name,
                    model=model
                )

                if success:
                    st.success("Transcription successful!")

                    # Display original text
                    st.subheader("Original Transcription:")
                    st.write(text)

                    # Automatic text processing
                    with st.spinner("Processing Text..."):
                        processed_text = self.text_processor.process_text(
                            text,
                            prompt,
                            model=chat_model
                        )
                        if processed_text:
                            st.subheader(f"Processed Text ({prompt.name}):")
                            st.write(processed_text)

                            # Download button for processed text
                            st.download_button(
                                label="Download Processed Text",
                                data=processed_text,
                                file_name="processed_text.txt",
                                mime="text/plain"
                            )
                else:
                    st.error(text)

                os.unlink(tmp_file.name)

    def handle_recording(self, audio_bytes, model, chat_model, prompt):
        with st.spinner("Processing Recording..."):
            tmp_file_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name

                text, success = self.transcriber.transcribe_file(
                    tmp_file_path,
                    model=model
                )

                if success:
                    st.success("Transcription successful!")

                    # Display original text
                    st.subheader("Original Transcription:")
                    st.write(text)

                    # Automatic text processing
                    with st.spinner("Processing Text..."):
                        processed_text = self.text_processor.process_text(
                            text,
                            prompt,
                            model=chat_model
                        )
                        if processed_text:
                            st.subheader(f"Processed Text ({prompt.name}):")
                            st.write(processed_text)
                else:
                    st.error(text)

            finally:
                if tmp_file_path:
                    max_retries = 3
                    for _ in range(max_retries):
                        try:
                            os.unlink(tmp_file_path)
                            break
                        except PermissionError:
                            time.sleep(0.1)

    def validate_text_input(self, text: str) -> tuple[bool, str]:
        """Validates the text input and returns (is_valid, message)"""
        if not text.strip():
            return False, "Text cannot be empty"
        if len(text) > 5000:  # Reasonable limit for API calls
            return False, "Text exceeds maximum length of 5000 characters"
        return True, ""

    def count_tokens(self, text: str) -> int:
        """Approximate token count for billing purposes"""
        # Rough approximation: 4 characters per token
        return len(text) // 4

if __name__ == "__main__":
    app = TranscriptionApp()
    app.setup_ui()