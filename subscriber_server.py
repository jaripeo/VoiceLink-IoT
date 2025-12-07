#!/usr/bin/env python3
"""
subscriber_server.py

MQTT subscriber that saves incoming WAV payloads and a small Flask webapp
that provides interactive playback + visualizations (WaveSurfer + Plotly).

Place this file in the same folder as `pitch_stft.py` and `basic_io.py`.
Run: `python subscriber_server.py` and open http://localhost:5000
"""
import io
import os
import threading
from pathlib import Path

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import paho.mqtt.client as mqtt

from basic_io import read_wav_mono, write_wav_mono
from pitch_stft import stft_pitch_shift, hilbert_pitch_shift
from plot_waveforms import plot_waveforms_to_buffer
import numpy as np
from scipy.signal import stft

# Configuration
# MQTT broker can be overridden via the environment variable `MQTT_BROKER`.
BROKER = os.environ.get("MQTT_BROKER", "172.20.10.4")
TOPIC = "audio/sample"
CONTROL_TOPIC = "audio/control"
RECEIVED = "received.wav"
PITCHED = "pitched.wav"
PITCHED_HILBERT = "pitched_hilbert.wav"
PITCHED_STFT = "pitched_stft.wav"
# Expected remote recording length (seconds) returned to frontend so it can show a timer
REMOTE_RECORD_SECONDS = int(os.environ.get("REMOTE_RECORD_SECONDS", "5"))

app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

# Thread-safety
file_lock = threading.Lock()


def safe_write(path: str, data: bytes):
    with file_lock:
        with open(path, "wb") as f:
            f.write(data)


def mqtt_on_message(client, userdata, msg):
    print("MQTT: received audio payload")
    # Save payload as WAV file
    try:
        safe_write(RECEIVED, msg.payload)
        print(f"Saved {RECEIVED}")
    except Exception as e:
        print("Error saving payload:", e)


mqtt_client = None


def start_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = mqtt_on_message
    try:
        print(f"Connecting to MQTT broker {BROKER}:1883")
        mqtt_client.connect(BROKER, 1883, 60)
    except Exception as e:
        print("Could not connect to MQTT broker:", e)
        return
    mqtt_client.subscribe(TOPIC)
    # Also subscribe to control topic only to show activity; the Pi listens for control.
    mqtt_client.subscribe(CONTROL_TOPIC)
    thread = threading.Thread(target=mqtt_client.loop_forever, daemon=True)
    thread.start()
    print("MQTT loop started (background)")
    print(f"Subscribed to topic '{TOPIC}' and control topic '{CONTROL_TOPIC}' (broker={BROKER})")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/audio/original")
def audio_original():
    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)
    return send_file(RECEIVED, mimetype="audio/wav")


@app.route("/audio/pitched")
def audio_pitched():
    """Return a pitched WAV (uses currently selected algorithm via query arg 'algorithm').
    Query args: factor (float), algorithm (hilbert|stft, default: stft), nfft (int, optional).
    """
    factor = float(request.args.get("factor", "1.0"))
    algorithm = request.args.get("algorithm", "stft")  # hilbert or stft
    nfft = int(request.args.get("nfft", "1024"))

    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    # Read original, process, and write pitched file under lock
    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        if algorithm == "hilbert":
            y = hilbert_pitch_shift(x, pitch_factor=factor)
            output_file = PITCHED_HILBERT
        else:  # stft
            y = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
            output_file = PITCHED_STFT
        write_wav_mono(output_file, y, sr)

    return send_file(output_file, mimetype="audio/wav")


@app.route("/audio/pitched-hilbert")
def audio_pitched_hilbert():
    """Return Hilbert-transformed pitched WAV."""
    factor = float(request.args.get("factor", "1.0"))

    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        y = hilbert_pitch_shift(x, pitch_factor=factor)
        write_wav_mono(PITCHED_HILBERT, y, sr)

    return send_file(PITCHED_HILBERT, mimetype="audio/wav")


@app.route("/audio/pitched-stft")
def audio_pitched_stft():
    """Return STFT-based pitched WAV."""
    factor = float(request.args.get("factor", "1.0"))
    nfft = int(request.args.get("nfft", "1024"))

    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        y = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
        write_wav_mono(PITCHED_STFT, y, sr)

    return send_file(PITCHED_STFT, mimetype="audio/wav")


def compute_spectrogram_array(x: np.ndarray, sr: int, nperseg=1024, max_shape=(300, 300)):
    # compute STFT magnitude (dB)
    f, t, Zxx = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, boundary='zeros')
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-9)

    # downsample to reasonable JSON size
    freq_bins, time_bins = S_db.shape
    max_freq, max_time = max_shape
    fi = np.linspace(0, freq_bins - 1, min(freq_bins, max_freq)).astype(int)
    ti = np.linspace(0, time_bins - 1, min(time_bins, max_time)).astype(int)
    S_small = S_db[np.ix_(fi, ti)]
    f_small = f[fi]
    t_small = t[ti]

    return f_small, t_small, S_small


@app.route("/api/waveform")
def api_waveform():
    """Return a downsampled waveform for plotting as JSON: {sr: int, samples: [..]}"""
    which = request.args.get("which", "original")
    if which == "original":
        path = RECEIVED
    else:
        # generate pitched in memory if requested
        factor = float(request.args.get("factor", "1.0"))
        nfft = int(request.args.get("nfft", "1024"))
        if not Path(RECEIVED).exists():
            return ("No audio received yet", 404)
        x, sr = read_wav_mono(RECEIVED)
        y = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
        samples = y
        # downsample for JSON
        max_points = 5000
        if len(samples) > max_points:
            idx = np.linspace(0, len(samples) - 1, max_points).astype(int)
            samples = samples[idx]
        return jsonify({"sr": sr, "samples": samples.tolist()})

    if not Path(path).exists():
        return ("No audio received yet", 404)
    x, sr = read_wav_mono(path)
    samples = x
    max_points = 5000
    if len(samples) > max_points:
        idx = np.linspace(0, len(samples) - 1, max_points).astype(int)
        samples = samples[idx]
    return jsonify({"sr": sr, "samples": samples.tolist()})


@app.route("/api/spectrogram")
def api_spectrogram():
    """Return spectrogram matrix as JSON: {f:[], t:[], S:[[...]]}.

    Query params:
      which: original | pitched | pitched-hilbert | pitched-stft (default: original)
      factor: pitch factor for pitched variants (default: 1.0)
      algorithm: hilbert | stft (optional; overrides when which=pitched)
      nperseg, max_freq, max_time: spectrogram sizing
    """
    which = request.args.get("which", "original")
    nperseg = int(request.args.get("nperseg", "1024"))
    max_freq = int(request.args.get("max_freq", "300"))
    max_time = int(request.args.get("max_time", "300"))

    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    x0, sr = read_wav_mono(RECEIVED)

    if which == "original":
        x = x0
    else:
        factor = float(request.args.get("factor", "1.0"))
        algo = request.args.get("algorithm")
        # normalize which -> algorithm choice
        if which == "pitched-hilbert":
            algo = "hilbert"
        elif which == "pitched-stft":
            algo = "stft"
        else:
            # generic "pitched" respects optional algorithm param, default stft
            algo = algo or "stft"

        if algo == "hilbert":
            x = hilbert_pitch_shift(x0, pitch_factor=factor)
        else:
            x = stft_pitch_shift(x0, sr, pitch_factor=factor, n_fft=nperseg)

    f_small, t_small, S_small = compute_spectrogram_array(x, sr, nperseg=nperseg, max_shape=(max_freq, max_time))

    return jsonify({
        "f": f_small.tolist(),
        "t": t_small.tolist(),
        "S": S_small.tolist(),
    })


# @app.route("/visuals/waveforms.png")
# def waveform_png():
#     """Return a PNG image with original and pitched waveforms.

@app.route('/control/record', methods=['POST'])
def control_record():
    """Publish a control message to request the Pi to record once.

    The route publishes the payload 'record' to the `audio/control` topic.
    """
    if mqtt_client is None:
        return ("MQTT client not connected", 500)
    try:
        # Publish control and return JSON with expected duration so frontend can show a timer
        mqtt_client.publish(CONTROL_TOPIC, 'record')
        print(f"Published control 'record' to topic '{CONTROL_TOPIC}' (broker={BROKER})")
        return jsonify({
            "status": "requested",
            "topic": CONTROL_TOPIC,
            "broker": BROKER,
            "expected_seconds": REMOTE_RECORD_SECONDS,
        }), 200
    except Exception as e:
        return (f"Failed to publish control: {e}", 500)


@app.route('/visuals/waveforms-all')
def waveforms_all():
    """Return PNG with all three waveforms: original, Hilbert pitched, STFT pitched.
    
    Query args:
      factor: pitch factor (float, default 1.0)
      duration: seconds to plot from start (float, default 0.1)
    """
    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    factor = float(request.args.get("factor", "1.0"))
    duration = float(request.args.get("duration", "1.0"))
    nfft = int(request.args.get("nfft", "1024"))

    # Generate both pitched versions under lock
    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        y_hilbert = hilbert_pitch_shift(x, pitch_factor=factor)
        y_stft = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
        write_wav_mono(PITCHED_HILBERT, y_hilbert, sr)
        write_wav_mono(PITCHED_STFT, y_stft, sr)

    # Create comparison waveform plot (original + both pitched)
    try:
        import matplotlib.pyplot as plt
        x_orig, sr_orig = read_wav_mono(RECEIVED)
        x_hilbert, _ = read_wav_mono(PITCHED_HILBERT)
        x_stft, _ = read_wav_mono(PITCHED_STFT)

        t_orig = np.arange(len(x_orig)) / sr_orig
        t_hilbert = np.arange(len(x_hilbert)) / sr_orig
        t_stft = np.arange(len(x_stft)) / sr_orig

        mask_orig = t_orig < duration
        mask_hilbert = t_hilbert < duration
        mask_stft = t_stft < duration

        plt.figure(figsize=(15, 9))

        plt.subplot(3, 1, 1)
        plt.plot(t_orig[mask_orig], x_orig[mask_orig], color='#2563eb', linewidth=0.8)
        plt.title('Original Waveform', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(t_hilbert[mask_hilbert], x_hilbert[mask_hilbert], color='#059669', linewidth=0.8)
        plt.title('Pitched (Hilbert)', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(t_stft[mask_stft], x_stft[mask_stft], color='#fb923c', linewidth=0.8)
        plt.title('Pitched (STFT)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return (f"Error generating waveform image: {e}", 500)


if __name__ == "__main__":
    start_mqtt()
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

