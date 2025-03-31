import streamlit as st
from speech_to_text import AudioTranscriber
from text_processors import TextProcessor
from prompts import AVAILABLE_PROMPTS, PromptTemplate
from streamlit_mic_recorder import mic_recorder
import tempfile
from dotenv import load_dotenv
import os
import time

load_dotenv()

class TranscriptionApp:
    def __init__(self):
        if 'provider' not in st.session_state:
            st.session_state.provider = 'groq'
        
        # Initialize all providers and cache their models at startup
        self.initialize_all_providers()
        
        # Setup the selected provider
        self.setup_provider()
    
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

    def setup_ui(self):
        st.title("AI Audio Transcription")
        
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
