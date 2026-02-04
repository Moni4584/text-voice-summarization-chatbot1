import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import tempfile
import os
import nltk

# Ensure punkt tokenizer exists (download only if missing)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

st.set_page_config(page_title="Text & Voice Summarization App", layout="centered")
st.title("📄 Text & Voice Summarization Chatbot")
st.write("Enter text or upload audio to get a summary.")

# -------- Text Summarization Function --------
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

# -------- Audio Summarization Function --------
def summarize_audio(audio_file):
    recognizer = sr.Recognizer()

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name

        # Try to convert uploaded audio to wav via pydub
        try:
            # Determine extension from filename if available
            filename = getattr(audio_file, "name", "")
            ext = filename.split(".")[-1].lower() if filename and "." in filename else None

            if ext and ext != "wav":
                # Read using pydub with explicit format
                audio = AudioSegment.from_file(audio_file, format=ext)
            else:
                # Let pydub detect format (may work for wav or file-like objects)
                audio = AudioSegment.from_file(audio_file)
            audio.export(temp_path, format="wav")
        except Exception:
            # Fallback: write raw bytes to temp file and hope it's already WAV
            audio_file.seek(0)
            temp_audio.write(audio_file.read())
            temp_audio.flush()

    # Transcribe with speech_recognition
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
        raise RuntimeError("Speech was unintelligible or could not be recognized.")
    except sr.RequestError as e:
        raise RuntimeError(f"Could not request results from speech recognition service: {e}")
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

    summary = summarize_text(text) if text else ""
    return text, summary


tab1, tab2 = st.tabs(["📝 Text Summarization", "🎤 Voice Summarization"])

# -------- TEXT TAB --------
with tab1:
    st.subheader("Enter Text to Summarize")
    text_input = st.text_area("Paste your text here:")

    if st.button("Summarize Text"):
        if text_input.strip():
            with st.spinner("Summarizing..."):
                summary = summarize_text(text_input)
            st.success("✅ Summary Generated")
            st.write("**Summary:**")
            st.write(summary)
        else:
            st.warning("Please enter some text.")


# -------- AUDIO TAB --------
with tab2:
    st.subheader("Upload Audio File (wav/mp3)")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

    if st.button("Summarize Audio"):
        if audio_file:
            with st.spinner("Transcribing and summarizing..."):
                try:
                    text, summary = summarize_audio(audio_file)
                    st.success("✅ Audio Processed")
                    st.write("**Transcription:**")
                    st.write(text)
                    st.write("**Summary:**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload an audio file.")
