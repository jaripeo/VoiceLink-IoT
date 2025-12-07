# Audio receiver + visualizer

This app receives WAV blobs over MQTT (topic `audio/sample`), saves them as `received.wav`, and serves a web UI to play original and pitch-shifted audio, plus spectrograms and waveform comparisons.

## Team members
- Dave Rodriguez
- Minjun Kim 

## Files
- `subscriber_server.py` — Flask app + MQTT subscriber; serves API + UI.
- `auto_pub.py` — record and publish .wav file from rpi to broker.
- `static/index.html` — frontend UI using Plotly for spectrograms and audio players.
- `basic_io.py` — WAV I/O utilities (normalize to/from float).
- `pitch_stft.py` — Hilbert and STFT pitch-shift implementations.
- `plot_waveforms.py` — renders waveform PNGs for comparison view.
- `requirements.txt` — Python dependencies.

## How to run (server + UI)
Run these commands on terminal <br>
- pip install -r requirements.txt <br>
- run "python subscriber_server.py" on laptop (broker IP should be laptop's IP Address on line 34) <br>
- run "python auto_pub.py" on RPI (broker IP should be laptop's IP Address on line 29) <br>

Then open http://localhost:5000 in your browser.

The user should see the frontend display to begin recording. 

## External libraries
- Flask, flask-cors
- paho-mqtt
- numpy, scipy (signal)
- matplotlib
- plotly (frontend)
