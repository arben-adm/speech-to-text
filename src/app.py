import streamlit as st
from src.speech_to_text import AudioTranscriber
from src.text_processors import TextProcessor
from src.prompts import AVAILABLE_PROMPTS, PromptTemplate
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
        
        self.setup_provider()
        
    def setup_provider(self):
        """Initializes the selected provider"""
        provider = st.session_state.provider.lower()
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        
        self.transcriber = AudioTranscriber(provider=provider, api_key=api_key)
        self.text_processor = TextProcessor(provider=provider, api_key=api_key)

    def setup_ui(self):
        st.title("AI Audio Transcription")
        
        # Provider selection
        col1, col2 = st.columns([2,1])
        with col1:
            provider = st.selectbox(
                "Select AI Provider:",
                options=['Groq', 'OpenAI'],
                index=0 if st.session_state.provider == 'groq' else 1
            )
            
            if provider.lower() != st.session_state.provider:
                st.session_state.provider = provider.lower()
                self.setup_provider()
                st.rerun()
                
        with col2:
            st.markdown(f"**Active Provider:** {provider}")
        
        # Model selection based on provider
        model = st.selectbox(
            "Select Model:",
            options=self.get_available_models()
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
        tab1, tab2 = st.tabs(["File Upload", "Microphone Recording"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Audio", 
                type=['mp3','wav','m4a']
            )
            if uploaded_file:
                self.handle_file_upload(uploaded_file, model, prompt)
                
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
                self.handle_recording(audio['bytes'], model, prompt)

    def get_available_models(self):
        """Returns available models based on the provider"""
        if st.session_state.provider == 'groq':
            return ["whisper-large-v3", "llama-3.3-70b-versatile"]
        else:
            return ["whisper-1", "gpt-4o-mini"]

    def handle_file_upload(self, uploaded_file, model, prompt):
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
                            prompt
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

    def handle_recording(self, audio_bytes, model, prompt):
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
                            prompt
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

if __name__ == "__main__":
    app = TranscriptionApp()
    app.setup_ui()
