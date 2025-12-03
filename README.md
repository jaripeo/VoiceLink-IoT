# Audio receiver + visualizer

This small app receives WAV blobs over MQTT (topic `audio/sample`), saves them as `received.wav`, and provides a small web UI to play the original and pitch-shifted audio and inspect spectrograms.

Files added in this folder:

- `subscriber_server.py` — Flask app + MQTT subscriber. Serves the frontend and provides API endpoints.
- `static/index.html` — frontend UI using WaveSurfer.js and Plotly.
- `requirements.txt` — Python dependencies (use conda recommended for numpy/scipy compatibility).

Quick start (recommended: conda)

```powershell
conda create -n audio_app python=3.11
conda activate audio_app
conda install -c conda-forge numpy=1.24.4 scipy=1.12.0
pip install -r requirements.txt
python subscriber_server.py
```

Then open http://localhost:5000 in your browser.

RPi side
- Use your existing `auto_publisher.py` on the Pi. It publishes raw WAV bytes to the MQTT broker and topic `audio/sample`.

Notes
- The Flask server generates pitched audio on demand using `stft_pitch_shift` from `pitch_stft.py` and `read_wav_mono` / `write_wav_mono` from `basic_io.py`.
- For spectrogram endpoints the server downsamples the spectrogram to keep JSON sizes reasonable.
