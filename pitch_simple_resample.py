# file: pitch_simple_resample.py
import numpy as np
from scipy.signal import resample
from basic_io import read_wav_mono, write_wav_mono

def pitch_shift_resample(x, pitch_factor):
    """
    Simple pitch shift by resampling.
    pitch_factor > 1.0  -> higher pitch, shorter duration
    pitch_factor < 1.0  -> lower pitch, longer duration
    """
    N = len(x)
    new_length = int(N / pitch_factor)  # shorter if pitch up
    y = resample(x, new_length)
    return y

if __name__ == "__main__":
    x, sr = read_wav_mono("test.wav")
    p = 1.3  # 30% higher pitch
    y = pitch_shift_resample(x, p)
    write_wav_mono("test_pitch_resample.wav", y, sr)
