import io
import numpy as np
import matplotlib.pyplot as plt
from basic_io import read_wav_mono


def plot_waveforms_to_buffer(original_path, pitched_path, duration=0.2):
    """Render original and pitched waveforms (first `duration` seconds) to a PNG buffer.

    Returns: BytesIO with PNG image (seeked to 0).
    """
    x, sr_x = read_wav_mono(original_path)
    y, sr_y = read_wav_mono(pitched_path)

    if sr_x != sr_y:
        raise ValueError("Sample rates must match")
    sr = sr_x

    t_x = np.arange(len(x)) / sr
    t_y = np.arange(len(y)) / sr

    mask_x = t_x < duration
    mask_y = t_y < duration

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_x[mask_x], x[mask_x])
    plt.title(f"Original waveform: {original_path}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(t_y[mask_y], y[mask_y])
    plt.title(f"Pitched waveform: {pitched_path}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


if __name__ == "__main__":
    # local test; change filenames if needed
    buf = plot_waveforms_to_buffer("test.wav", "pitched.wav")
    with open("waveforms_debug.png", "wb") as f:
        f.write(buf.getvalue())
    print("Wrote waveforms_debug.png")

if __name__ == "__main__":
    # Change these to your filenames
    plot_waveforms("test.wav", "pitched.wav")


# plot_waveforms.py
import numpy as np
import matplotlib.pyplot as plt
from basic_io import read_wav_mono

def plot_waveforms(original_path, pitched_path):
    x, sr_x = read_wav_mono(original_path)
    y, sr_y = read_wav_mono(pitched_path)

    assert sr_x == sr_y, "Sample rates must match"
    sr = sr_x

    # Make a time axis for each
    t_x = np.arange(len(x)) / sr
    t_y = np.arange(len(y)) / sr

    # For clarity, just look at first 0.1 seconds (adjust if you want)
    duration = 0.1
    mask_x = t_x < duration
    mask_y = t_y < duration

    plt.figure(figsize=(10, 6))

    # Original
    plt.subplot(2, 1, 1)
    plt.plot(t_x[mask_x], x[mask_x])
    plt.title(f"Original waveform: {original_path}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Pitched
    plt.subplot(2, 1, 2)
    plt.plot(t_y[mask_y], y[mask_y])
    plt.title(f"Pitched waveform: {pitched_path}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change these to your filenames
    plot_waveforms("test.wav", "pitched.wav")

