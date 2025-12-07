
import numpy as np
from scipy.io import wavfile

def read_wav_mono(path):
    """
    Reads a WAV file using scipy.io.wavfile.
    Supports 16-bit, 24-bit (packed in int32), 32-bit float, etc.
    Returns (audio_float, sample_rate) where audio_float is in [-1, 1].
    """
    sr, data = wavfile.read(path)  # data: np.ndarray

    # Downmix stereo to mono if needed
    if data.ndim == 2:
        # take left channel
        data = data[:, 0]

    # Convert to float in [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0

    elif data.dtype == np.int32:
        # Often used for 24-bit packed in 32 bits
        audio = data.astype(np.float32) / 2147483648.0  # 2^31

    elif data.dtype == np.uint8:
        # 8-bit PCM is unsigned [0,255] -> [-1,1]
        audio = (data.astype(np.float32) - 128.0) / 128.0

    elif data.dtype in (np.float32, np.float64):
        # Already float; just cast and clip
        audio = data.astype(np.float32)
        # Ensure in [-1,1]
        max_abs = np.max(np.abs(audio)) + 1e-9
        if max_abs > 1.0:
            audio = audio / max_abs

    else:
        raise ValueError(f"Unsupported WAV data type: {data.dtype}")

    return audio, sr


def write_wav_mono(path, audio_float, sample_rate):
    """
    Writes mono float audio in [-1,1] to a 16-bit PCM WAV.
    """
    from scipy.io import wavfile

    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    wavfile.write(path, sample_rate, audio_int16)

