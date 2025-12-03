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
from pitch_stft import stft_pitch_shift
from plot_waveforms import plot_waveforms_to_buffer
import numpy as np
from scipy.signal import stft

# Configuration
# MQTT broker can be overridden via the environment variable `MQTT_BROKER`.
BROKER = os.environ.get("MQTT_BROKER", "172.20.10.2")
TOPIC = "audio/sample"
CONTROL_TOPIC = "audio/control"
RECEIVED = "received.wav"
PITCHED = "pitched.wav"

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
    """Return a pitched WAV. Query args: factor (float), nfft (int, optional).
    This generates the pitched audio on demand from the latest received file.
    """
    factor = float(request.args.get("factor", "1.0"))
    nfft = int(request.args.get("nfft", "1024"))

    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    # Read original, process, and write pitched file under lock
    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        y = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
        write_wav_mono(PITCHED, y, sr)

    return send_file(PITCHED, mimetype="audio/wav")


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
    """Return a small spectrogram matrix as JSON: {f:[], t:[], S: [[...]]}
    Query: which=original|pitched, factor (if pitched), nperseg
    """
    which = request.args.get("which", "original")
    nperseg = int(request.args.get("nperseg", "1024"))
    max_freq = int(request.args.get("max_freq", "300"))
    max_time = int(request.args.get("max_time", "300"))

    if which == "original":
        if not Path(RECEIVED).exists():
            return ("No audio received yet", 404)
        x, sr = read_wav_mono(RECEIVED)
    else:
        # pitched
        factor = float(request.args.get("factor", "1.0"))
        if not Path(RECEIVED).exists():
            return ("No audio received yet", 404)
        x0, sr = read_wav_mono(RECEIVED)
        x = stft_pitch_shift(x0, sr, pitch_factor=factor, n_fft=nperseg)

    f_small, t_small, S_small = compute_spectrogram_array(x, sr, nperseg=nperseg, max_shape=(max_freq, max_time))

    return jsonify({
        "f": f_small.tolist(),
        "t": t_small.tolist(),
        "S": S_small.tolist(),
    })


@app.route("/visuals/waveforms.png")
def waveform_png():
    """Return a PNG image with original and pitched waveforms.

    Query args:
      factor: pitch factor for pitched waveform (float, optional)
      duration: seconds to plot from start (float, optional)
    """
    if not Path(RECEIVED).exists():
        return ("No audio received yet", 404)

    factor = float(request.args.get("factor", "1.0"))
    duration = float(request.args.get("duration", "0.1"))
    nfft = int(request.args.get("nfft", "1024"))

    # Ensure pitched file exists (generate under lock)
    with file_lock:
        x, sr = read_wav_mono(RECEIVED)
        y = stft_pitch_shift(x, sr, pitch_factor=factor, n_fft=nfft)
        write_wav_mono(PITCHED, y, sr)

    # Use helper to create PNG buffer
    try:
        buf = plot_waveforms_to_buffer(RECEIVED, PITCHED, duration=duration)
    except Exception as e:
        return (f"Error generating waveform image: {e}", 500)

    return send_file(buf, mimetype="image/png")


@app.route('/control/record', methods=['POST'])
def control_record():
    """Publish a control message to request the Pi to record once.

    The route publishes the payload 'record' to the `audio/control` topic.
    """
    if mqtt_client is None:
        return ("MQTT client not connected", 500)
    try:
        mqtt_client.publish(CONTROL_TOPIC, 'record')
        print(f"Published control 'record' to topic '{CONTROL_TOPIC}' (broker={BROKER})")
        return ("Requested remote recording", 200)
    except Exception as e:
        return (f"Failed to publish control: {e}", 500)


if __name__ == "__main__":
    start_mqtt()
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

