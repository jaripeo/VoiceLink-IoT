# file: basic_io.py
import wave
import numpy as np

def read_wav_mono(path):
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    # Convert to numpy int16
    audio = np.frombuffer(frames, dtype=np.int16)

    # If stereo, take one channel
    if num_channels == 2:
        audio = audio[0::2]

    # Normalize to float32 in [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    return audio, framerate

def write_wav_mono(path, audio_float, samplerate):
    # Clip and convert back to int16
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    x, sr = read_wav_mono("test.wav")
    print("Loaded:", x.shape, "sample rate:", sr)
    write_wav_mono("test_copy.wav", x, sr)
