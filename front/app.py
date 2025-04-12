import streamlit as st
import requests
import io # Needed for sending bytes as files

# --- Configuration ---
# !!! IMPORTANT: Replace this with the actual URL of your backend API !!!
BACKEND_CLONING_URL = "http://localhost:8000/clone_voice" # Example URL

st.set_page_config(layout="wide")

st.title("Voice Cloning")

# Initialize variables to store data and types
input1_data = None
input1_type = None
voice_ref_data = None
voice_ref_type = None

# --- Section 1: Input Source ---
st.subheader("1. Input Source (Content to Generate)")
input1_choice = st.radio(
    "Choose the type of input content:",
    ("Text", "Upload MP3 File", "Record Audio"),
    key="input1_choice_radio"
)

if input1_choice == "Text":
    input1_data = st.text_area(
        "Enter the text you want the cloned voice to speak:",
        height=150,
        placeholder="Type your text here...",
        key="input1_text"
    )
    input1_type = "text"
    if input1_data:
        st.success("Text input provided.")

elif input1_choice == "Upload MP3 File":
    uploaded_file_input1 = st.file_uploader(
        "Upload an MP3 file as input content:",
        type=['mp3'],
        key="input1_uploader"
    )
    if uploaded_file_input1 is not None:
        input1_data = uploaded_file_input1.getvalue()
        input1_type = "mp3"
        st.audio(input1_data, format='audio/mp3')
        st.success("Input MP3 uploaded successfully.")

elif input1_choice == "Record Audio":
    input1_data = st.audio_input(
        "Record the input content:",
        key="input1_recorder"
    )
    if input1_data is not None:
        input1_type = "audio_bytes"
        st.success("Input audio recorded successfully.")

st.divider()

# --- Section 2: Voice Reference ---
st.subheader("2. Voice Reference (Voice to Clone)")
voice_ref_choice = st.radio(
    "Choose the type of voice reference:",
    ("Upload MP3 File", "Record Audio"),
    key="voice_ref_choice_radio"
)

if voice_ref_choice == "Upload MP3 File":
    uploaded_file_ref = st.file_uploader(
        "Upload an MP3 file for the voice reference:",
        type=['mp3', 'wav'],  # Include wav files
        key="ref_uploader"
    )
    if uploaded_file_ref is not None:
        voice_ref_data = uploaded_file_ref.getvalue()
        voice_ref_type = uploaded_file_ref.type.split('/')[1]  # Determine type based on uploaded file
        st.audio(voice_ref_data, format=uploaded_file_ref.type)  # Use the correct format
        st.success(f"Voice reference {voice_ref_type.upper()} uploaded successfully.")

elif voice_ref_choice == "Record Audio":
    voice_ref_data = st.audio_input(
        "Record the voice reference:",
        key="ref_recorder"
    )
    if voice_ref_data is not None:
        voice_ref_type = "audio_bytes"
        st.success("Voice reference recorded successfully.")

st.divider()

# --- Section 3: Cloning Process ---
st.subheader("3. Start Cloning")

# Check if both inputs are provided
input1_provided = False
if input1_type == "text":
    input1_provided = input1_data is not None and input1_data.strip() != ""
else:
    input1_provided = input1_data is not None

voice_ref_provided = voice_ref_data is not None

if input1_provided and voice_ref_provided:
    if st.button("Start Cloning Process"):
        with st.spinner("Sending data and cloning voice... Please wait."):
            try:
                # Prepare data payload (multipart/form-data)
                files_payload = {}
                data_payload = {
                    'input1_type': input1_type,
                    'voice_ref_type': voice_ref_type
                }

                # Add Input 1 data
                if input1_type == 'text':
                    data_payload['input1_text'] = input1_data
                elif input1_type == 'mp3':
                    files_payload['input1_audio'] = ('input1.mp3', input1_data, 'audio/mpeg')
                elif input1_type == 'audio_bytes': # Assuming recorder gives wav
                    files_payload['input1_audio'] = ('input1.wav', input1_data, 'audio/wav')

                # Add Voice Reference data
                try:
                    if voice_ref_type == 'mp3':
                        files_payload['voice_ref_audio'] = ('voice_ref.mp3', voice_ref_data, 'audio/mpeg')
                    elif voice_ref_type == 'audio_bytes': # Assuming recorder gives wav
                        files_payload['voice_ref_audio'] = ('voice_ref.mp3', voice_ref_data, 'audio/wav')
                except Exception as e:
                    st.error(f"Error processing voice reference: {e}")
                response = requests.post(
                    BACKEND_CLONING_URL,
                    files=files_payload,
                    data=data_payload
                )

                # Handle response
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                # Check if response contains audio
                content_type = response.headers.get('Content-Type', '').lower()
                if 'audio' in content_type:
                    st.success("Voice cloning successful!")
                    st.audio(response.content, format=content_type)
                    st.balloons()
                else:
                    # If no audio, maybe backend sent a JSON message?
                    try:
                        error_info = response.json()
                        st.error(f"Cloning failed on backend: {error_info.get('detail', response.text)}")
                    except requests.exceptions.JSONDecodeError:
                        st.error(f"Cloning failed. Received unexpected response: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Network error: Failed to connect to the backend. {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    warning_message = "Please provide both:"
    if not input1_provided:
        warning_message += "\n - Input Source (Section 1)"
    if not voice_ref_provided:
        warning_message += "\n - Voice Reference (Section 2)"
    st.warning(warning_message)
    # Disable the button explicitly if needed, though Streamlit typically handles this
    st.button("Start Cloning Process", disabled=True)
