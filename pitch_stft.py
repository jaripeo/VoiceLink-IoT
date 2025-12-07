
import argparse
import numpy as np
from scipy.signal import stft, istft, hilbert
from basic_io import read_wav_mono, write_wav_mono

def hilbert_pitch_shift(x, pitch_factor):
    """
    Pitch shift using analytic signal (Hilbert transform),
    with phase scaling anchored at the initial phase to better
    preserve overall waveform shape.

    pitch_factor > 1.0 -> pitch UP
    pitch_factor < 1.0 -> pitch DOWN
    """
    # Remove DC to avoid weird envelope behavior
    x_dc = x - np.mean(x)

    # Analytic signal: x_a = a(t) * exp(j * phi(t))
    analytic = hilbert(x_dc)
    amp = np.abs(analytic)                      # envelope a(t)
    phase = np.unwrap(np.angle(analytic))       # continuous phase phi(t)

    # Anchor phase at the first sample so we don't globally flip shape
    phi0 = phase[0]
    phase_centered = phase - phi0
    phase_new = phi0 + pitch_factor * phase_centered

    # Reconstruct real signal with same envelope but modified phase
    y = amp * np.cos(phase_new)

    # Restore DC level approximately
    y = y + np.mean(x)

    # Normalize gently to avoid clipping (only if needed)
    max_abs = np.max(np.abs(y)) + 1e-9
    if max_abs > 1.0:
        y = y / max_abs

    return y.astype(np.float32)



def stft_pitch_shift(x, sr, pitch_factor, n_fft=2048, hop_length=None):
    """
    Simple STFT-based pitch shifter (educational version).

    pitch_factor > 1.0 -> pitch UP (higher)
    pitch_factor < 1.0 -> pitch DOWN (lower)

    Steps:
      1) STFT (windowed FFT over time)
      2) Scale spectrum along frequency axis
      3) iSTFT to reconstruct time signal
    """
    if hop_length is None:
        hop_length = n_fft // 4  # typical choice = 75% overlap

    # STFT: Zxx has shape (num_freq_bins, num_frames)
    f, t, Zxx = stft(
        x,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        padded=True,
        boundary="zeros"
    )

    num_bins, num_frames = Zxx.shape
    Z_shifted = np.zeros_like(Zxx, dtype=np.complex64)

    # For each output bin, choose a source bin index.
    # We want frequencies multiplied by pitch_factor:
    #   f_out = pitch_factor * f_in
    #   => bin_out = pitch_factor * bin_in
    #   => bin_in = bin_out / pitch_factor
    for out_bin in range(num_bins):
        src_bin = int(out_bin / pitch_factor)  # nearest neighbor
        if 0 <= src_bin < num_bins:
            Z_shifted[out_bin, :] = Zxx[src_bin, :]

    # Inverse STFT
    _, x_recon = istft(
        Z_shifted,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        input_onesided=True
    )

    # Match original length (trim or pad)
    if len(x_recon) > len(x):
        x_recon = x_recon[:len(x)]
    else:
        pad = len(x) - len(x_recon)
        if pad > 0:
            x_recon = np.pad(x_recon, (0, pad))

    # Normalize a bit to avoid clipping
    max_abs = np.max(np.abs(x_recon)) + 1e-9
    x_recon = x_recon / max_abs * min(1.0, max_abs)

    return x_recon.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Dual pitch shifter: compare Hilbert vs STFT methods."
    )
    parser.add_argument("input", help="Input WAV file (mono 16-bit PCM)")
    parser.add_argument(
        "--factor",
        type=float,
        default=1.2,
        help="Pitch factor (>1 up, <1 down), e.g. 1.2 or 0.8"
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=2048,
        help="FFT size for STFT (default 2048)"
    )
    parser.add_argument(
        "--method",
        choices=["hilbert", "stft", "both"],
        default="both",
        help="Which method to use (default: both)"
    )

    args = parser.parse_args()

    x, sr = read_wav_mono(args.input)
    print(f"Loaded {args.input}, {len(x)} samples at {sr} Hz")

    if args.method in ("hilbert", "both"):
        y_hilbert = hilbert_pitch_shift(x, args.factor)
        out_hilbert = args.input.rsplit('.', 1)[0] + "_hilbert.wav"
        write_wav_mono(out_hilbert, y_hilbert, sr)
        print(f"Saved Hilbert pitch-shifted audio to {out_hilbert}")

    if args.method in ("stft", "both"):
        y_stft = stft_pitch_shift(x, sr, args.factor, n_fft=args.nfft)
        out_stft = args.input.rsplit('.', 1)[0] + "_stft.wav"
        write_wav_mono(out_stft, y_stft, sr)
        print(f"Saved STFT pitch-shifted audio to {out_stft}")


if __name__ == "__main__":
    main()