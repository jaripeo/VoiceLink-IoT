# pitch_stft.py
import argparse
import numpy as np
from scipy.signal import stft, istft
from basic_io import read_wav_mono, write_wav_mono

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
        description="Simple STFT-based pitch shifter."
    )
    parser.add_argument("input", help="Input WAV file (mono 16-bit PCM)")
    parser.add_argument("output", help="Output WAV file")
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
        help="FFT size (window length), e.g. 1024 or 2048"
    )

    args = parser.parse_args()

    x, sr = read_wav_mono(args.input)
    print(f"Loaded {args.input}, {len(x)} samples at {sr} Hz")

    y = stft_pitch_shift(x, sr, pitch_factor=args.factor, n_fft=args.nfft)

    write_wav_mono(args.output, y, sr)
    print(f"Saved pitch-shifted audio to {args.output}")

if __name__ == "__main__":
    main()

