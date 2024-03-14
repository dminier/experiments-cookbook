import io
import os

import requests
import streamlit as st
from pydub import AudioSegment


def main():
    st.title('🎤 Transcribe Wave')
    prompt = st.text_input("Whisper prompt", value="<|startoftranscript|><|fr|><|transcribe|><|notimestamps|>")
    file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    api_http_url = os.getenv("API_HTTP_URL", "http://localhost:7000/rest/speech2text")

    with st.sidebar:
        chunk_size_ms = st.number_input('Chunk Size (ms)', value=10000)

    if file is not None:

        st.write(file.type)
        bytes_data = file.read()
        st.audio(bytes_data, format="audio")

        if st.button("Convert to WAV"):
            wav_bytes = convert(bytes_data)
            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))

            total_duration_ms = len(audio)
            st.write(f"Total duration :{total_duration_ms}")
            num_chunks = total_duration_ms // chunk_size_ms + 1
            st.write(f"Uploading {num_chunks} chunks...")
            st.session_state['transcription'] = ""
            result_container = st.container(height=600, border=2)

            for i in range(num_chunks):
                start_time = i * chunk_size_ms
                end_time = min((i + 1) * chunk_size_ms, total_duration_ms)

                chunk = audio[start_time:end_time]
                with io.BytesIO() as wav_chunk:
                    chunk.export(wav_chunk, format="wav")
                    files = {"file": wav_chunk.read()}

                try:
                    response = requests.post(api_http_url, files=files, data={"prompt": prompt})
                    st.session_state['transcription'] = response.text
                    result_container.write("".join(st.session_state['transcription']))
                except Exception as e:
                    st.error(f"Error occurred: {e}")


def convert(file_bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    st.write(f'convert to rate {audio.frame_rate}')
    with io.BytesIO() as wav_bytes:
        audio.export(wav_bytes, format="wav")
        return wav_bytes.getvalue()


if __name__ == "__main__":
    main()
