import io
import os
import time

import requests
import streamlit as st
from pydub import AudioSegment


def main():
    st.title('🎤 Transcribe file')
    prompt = st.text_input("Whisper prompt", value="<|startoftranscript|><|fr|><|transcribe|><|notimestamps|>")
    file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    api_http_url = os.getenv("API_HTTP_URL", "http://localhost:7000")

    with st.sidebar:
        channel_number = st.number_input('channel_number', value=-1)

    if file is not None:
        st.write(file.type)
        bytes_data = file.read()
        st.audio(bytes_data, format="audio")

        transcribe_no_timestamp(api_http_url, bytes_data, channel_number)
        transcribe_timestamp(api_http_url, bytes_data, channel_number)


def transcribe_no_timestamp(api_http_url, bytes_data, channel_number):


    if st.button("Transcribe no timestamp"):
        audio = AudioSegment.from_file(io.BytesIO(bytes_data))

        transcription = call(endpoint=f"{api_http_url}/rest/transcribe-simple",
                             file=bytes_data,
                             data={"language_code": 'fr',
                                   "channel_number" : channel_number})

        with st.expander("See Json"):
            st.write(transcription)
        with st.container(height=600):
            channels_data = split_by_channel_name(transcription)
            column = st.columns(len(channels_data))
            i = 0
            for channel, scripts in channels_data.items():
                with column[i]:
                    for chunk in scripts:
                        st.write(chunk['text'])
                i += 1

def transcribe_timestamp(api_http_url, bytes_data, channel_number):
    if st.button("Transcribe with timestamp"):
        audio = AudioSegment.from_file(io.BytesIO(bytes_data))

        transcription = call(endpoint=f"{api_http_url}/rest/transcribe-with-sentence-timestamp",
                             file=bytes_data,
                             data={"language_code": 'fr'})

        with st.expander("See Json"):
            st.write(transcription)
        with st.container(height=600):
            channels_data = split_by_channel_name(transcription)
            column = st.columns(len(channels_data))
            i = 0
            for channel, scripts in channels_data.items():
                with column[i]:
                    for chunk in scripts:
                        st.write(chunk['text'])
                i += 1

def split_by_channel_name(transcription):
    channel_data = {}

    # Parcourir chaque élément de la liste de chunks
    for chunk in transcription['chunks']:
        # Extraire le nom du canal
        channel_name = chunk['channel_name']
        # Vérifier si le nom du canal est déjà dans le dictionnaire
        if channel_name in channel_data:
            # Si oui, ajouter le chunk à la liste correspondante
            channel_data[channel_name].append(chunk)
        else:
            # Sinon, créer une nouvelle liste avec ce chunk
            channel_data[channel_name] = [chunk]

    return channel_data


def call(endpoint, file, data):
    start = time.time()
    response = requests.post(endpoint,
                             files={"file": ("filename", file, "audio/x-wav")},
                             data=data
                             )
    end = time.time()
    transcription = response.json()
    duration = end - start
    audio_duration = response.json()["audio_duration"]
    st.write(f"Audio Duration= {audio_duration} seconds")
    st.write(f"Time to transcribe = {duration} seconds")
    rtf = duration / audio_duration
    st.write(f"RTF = {rtf}")
    return transcription


if __name__ == "__main__":
    main()
