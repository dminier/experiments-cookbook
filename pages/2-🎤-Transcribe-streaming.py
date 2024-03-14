import asyncio
import base64
import os

import aiohttp
import pyaudio
import streamlit as st
import webrtcvad
from aiohttp import ClientConnectorError
from loguru import logger

API_WS_URL = os.getenv("API_WS_URL", "ws://localhost:7000")

if 'listen' not in st.session_state:
    st.session_state['listen'] = False
    st.session_state['transcription'] = ""

st.title('🎤 Real-Time Transcription App')

vad = webrtcvad.Vad()
vad.set_mode(1)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)


def start_listening():
    st.session_state['listen'] = True


def stop_listening():
    st.session_state['listen'] = False


def update_vad():
    vad.set_mode(VAD_SENSIBILITY)


with st.sidebar:
    st.button('Start', on_click=start_listening)
    st.button('Stop', on_click=stop_listening)
    VAD_SENSIBILITY = st.selectbox('VAD Sensibility', (0, 1, 2, 3), on_change=update_vad)
    NUM_PADDING_CHUNKS = st.number_input('Insert number of padding chunks', value=10)
    NUM_WINDOW_CHUNKS = st.number_input('Insert number of chunks in the windows', value=300)
    MINIMAL_SPEECH_IN_SECOND = st.number_input('Minimal audio duration before sens', value=0.5)
    RECORD_SECONDS = st.number_input('Max record in seconds', value=3.00)

result_container = st.container(height=600, border=2)

p = pyaudio.PyAudio()

# Open an audio stream with above parameter settings
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)


#######################################################################################################################
async def send(ws):
    num_vad_chunks = num_voiced_chunks = 0
    recorded_frames = []
    recording = False
    have_to_send = False
    while st.session_state['listen']:

        chunk = stream.read(CHUNK_SIZE)
        is_speech = vad.is_speech(chunk, RATE)

        # record when a sound (not really is_speech)
        if is_speech:
            num_voiced_chunks += 1
            recorded_frames.append(chunk)

            if not recording:
                logger.debug("Start recording ...")
                recording = True
        else:
            # If number of silent chunck >
            num_vad_chunks += 1
            if recording and num_vad_chunks > NUM_PADDING_CHUNKS + NUM_WINDOW_CHUNKS:
                recording = False

        if recording:
            recorded_frames.append(chunk)

        nb_frames = len(recorded_frames)
        duration = len(recorded_frames) * CHUNK_DURATION_MS * 1000

        if not recording and duration > MINIMAL_SPEECH_IN_SECOND:
            have_to_send = True

        if nb_frames >= NUM_WINDOW_CHUNKS or have_to_send:
            data = b''.join(recorded_frames)
            data = base64.b64encode(data).decode("utf-8")

            json_data = {"audio_data": str(data)}
            await ws.send_json(json_data)
            recorded_frames = []
            have_to_send = False

        await asyncio.sleep(0.01)


async def receive(ws):
    while st.session_state['listen']:
        message = await ws.receive_json()
        result = message['text']

        st.session_state['transcription'] = result
        result_container.write(st.session_state['transcription'])
        await asyncio.sleep(0.1)


async def send_receive():
    url = f"{API_WS_URL}/ws/speech2text"
    logger.debug(f'Connecting websocket to url {url}')
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.ws_connect(url) as ws:
                await asyncio.gather(send(ws), receive(ws))
    except ClientConnectorError:
        st.error('You should install and run https://github.com/dminier/whisper-triton-api', icon="🚨")


asyncio.run(send_receive())
