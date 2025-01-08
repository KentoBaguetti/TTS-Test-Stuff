import os
import openai
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import wave
import tempfile
from silero_vad import get_speech_timestamps, read_audio, collect_chunks
import torch
import time
from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice
import re

load_dotenv()

if __name__ == "__main__":
    print("Starting translator application...")

    # Set up OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print(
            "Error: OpenAI API key is not set. Make sure it's defined in the .env file."
        )
        exit(1)
    print("OpenAI API key loaded successfully.")

    languages = [
        ["english", "en"],
        ["german", "de"],
        ["french", "fr"],
        ["spanish", "es"],
        ["portuguese", "pt"],
        ["italian", "it"],
    ]

    # Load Silero VAD model
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    cache = {}

    def sanitize_text(text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-zA-Z0-9\s\.,\?!]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def generate_response(messages):
        """Generate assistant's response using OpenAI."""
        print("Sending request to OpenAI for translation...")
        try:
            for chunk in openai.ChatCompletion.create(
                model="gpt-4o-mini", messages=messages, stream=True
            ):
                text_chunk = chunk["choices"][0]["delta"].get("content")
                if text_chunk:
                    yield text_chunk
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            raise

    def clear_console():
        os.system("clear" if os.name == "posix" else "cls")

    def select_language():
        """Display language options and get user's choice."""
        print("Displaying language options:")
        for index, language in enumerate(languages, start=1):
            print(f"{index}. {language[0]}")
        language_number = input("Select language to translate to (1-6): ")
        selected_language = languages[int(language_number) - 1]
        print(f"Selected language: {selected_language[0]}")
        return selected_language

    def record_audio_vad():
        """Record audio using Silero VAD for continuous speech detection."""
        fs = 16000  # Sample rate
        duration = 10  # Buffer duration for listening in seconds
        print("Recording audio using Silero VAD...")
        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype="int16") as stream:
                buffer = []
                for _ in range(int(duration * fs / 512)):
                    frame = stream.read(512)[0]
                    buffer.extend(frame.flatten())
                audio_data = np.array(buffer, dtype="int16")
        except Exception as e:
            print(f"Error during audio recording: {e}")
            raise

        # Use Silero VAD to detect speech
        audio_tensor = (
            torch.tensor(audio_data, dtype=torch.float32) / 32768.0
        )  # Normalize to -1 to 1
        timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=fs)
        if not timestamps:
            print("No speech detected.")
            return None

        # Extract speech segments
        speech_data = collect_chunks(timestamps, audio_tensor)

        # Save the detected speech to a temporary file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes((speech_data * 32768).numpy().astype("int16").tobytes())
        print(f"Audio saved to temporary file: {temp_wav.name}")
        return temp_wav.name

    def transcribe_audio(file_path):
        """Transcribe audio using OpenAI Whisper."""
        print(f"Transcribing audio file: {file_path}")
        try:
            with open(file_path, "rb") as audio_file:
                response = openai.Audio.transcribe("whisper-1", audio_file)
            transcript = response.get("text", "")
            print(f"Transcription successful: {transcript}")
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def play_translation_with_gtts(
        translation_text, language_code, translation_start_time
    ):
        """Generate and play translation using GTTSEngine."""
        try:
            print("Initializing gTTS engine...")
            voice = GTTSVoice(language=language_code)
            engine = GTTSEngine(voice)
            stream = TextToAudioStream(engine)

            # Time when TTS starts
            tts_start_time = time.time()
            elapsed_time = tts_start_time - translation_start_time
            print(f"Time from translation to TTS start: {elapsed_time:.2f} seconds")

            print("Feeding text to TTS engine...")
            stream.feed(translation_text)
            print("Playing translation...")
            stream.play()
        except Exception as e:
            print(f"Error during TTS playback: {e}")

    def main():
        """Main translation loop."""
        clear_console()
        print("Starting main translation loop.")
        language_info = select_language()

        system_prompt_message = {
            "role": "system",
            "content": f"Translate the given text to {language_info[0]}. Output only the translated text.",
        }

        while True:
            print("\nSay something!")

            try:
                # Capture and transcribe user input
                audio_file = record_audio_vad()
                if not audio_file:
                    print("No valid speech detected. Please try again.")
                    continue
                user_text = transcribe_audio(audio_file)
                print(f"Input text: {user_text}")

                user_message = {"role": "user", "content": user_text}

                # Get translation
                print("Generating translation...")
                translation_text = ""
                translation_start_time = (
                    time.time()
                )  # Start timing for translation-to-TTS
                for chunk in generate_response([system_prompt_message, user_message]):
                    print(chunk, end="", flush=True)
                    translation_text += chunk

                print("\nPlaying translation...")
                if translation_text.strip():
                    play_translation_with_gtts(
                        translation_text, language_info[1], translation_start_time
                    )
                else:
                    print("No valid translation received.")
            except Exception as e:
                print(f"An error occurred: {e}")

    main()
