# plot_spectrograms.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from basic_io import read_wav_mono

def plot_spectros(original_path, pitched_path):
    x, sr_x = read_wav_mono(original_path)
    y, sr_y = read_wav_mono(pitched_path)
    assert sr_x == sr_y
    sr = sr_x

    f_x, t_x, Sx = spectrogram(x, fs=sr, nperseg=1024, noverlap=512)
    f_y, t_y, Sy = spectrogram(y, fs=sr, nperseg=1024, noverlap=512)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.pcolormesh(t_x, f_x, 10 * np.log10(Sx + 1e-10))
    plt.title(f"Original spectrogram: {original_path}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")

    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_y, f_y, 10 * np.log10(Sy + 1e-10))
    plt.title(f"Pitched spectrogram: {pitched_path}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_spectros("test.wav", "pitched.wav")

