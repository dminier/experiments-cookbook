from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from six import BytesIO

from whisper_spec import WhisperWaveSpec


def remove_silent(bytes_content):
    sound = AudioSegment.from_raw(BytesIO(bytes_content), sample_width=WhisperWaveSpec.BITS_PER_SAMPLE // 8,
                                  frame_rate=WhisperWaveSpec.FRAME_RATE,
                                  channels=WhisperWaveSpec.CHANNELS)
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    trimmed_sound = sound[start_trim:duration - end_trim]
    if len(sound) > 0.5:
        return trimmed_sound.raw_data
    else:
        return None
