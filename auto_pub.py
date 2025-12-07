
"""
auto_pub_control.py

Raspberry Pi publisher that records audio on-demand via MQTT control messages.

Usage:
  # subscribe to control messages and record when a 'record' payload arrives
  python3 auto_pub_control.py

  # run in automatic mode (old behavior)
  python3 auto_pub_control.py --auto

Notes:
 - Listens on topic `audio/control` for the payload `record` to trigger a single recording.
 - Publishes raw WAV bytes to `audio/sample` when a recording completes.
"""
import argparse
import time
import paho.mqtt.client as mqtt
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

BROKER = "172.20.10.4"  # change to your laptop/broker IP
TOPIC_SAMPLE = "audio/sample"
TOPIC_CONTROL = "audio/control"

SAMPLE_RATE = 16000
DURATION = 5


def record_once(sample_rate=SAMPLE_RATE, duration=DURATION, path="temp.wav"):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    #boost loudness
    gain = 3.0 #try 1.5 - 3.0; keep low enough to avoid clipping noise
    audio = np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

    print("Recording finished.")
    write(path, sample_rate, audio)
    with open(path, 'rb') as f:
        data = f.read()
    return data


def run_auto_mode(client):
    print("Running in automatic mode: recording every", DURATION, "s")
    try:
        while True:
            data = record_once()
            client.publish(TOPIC_SAMPLE, data)
            print("Published audio sample.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping auto mode")


def on_control_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8', errors='ignore').strip().lower()
    print("Control message received:", payload)
    if payload == 'record':
        try:
            data = record_once()
            client.publish(TOPIC_SAMPLE, data)
            print("Published audio sample (on-demand)")
        except Exception as e:
            print("Error recording/publishing:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--broker', default=BROKER, help='MQTT broker IP')
    parser.add_argument('--auto', action='store_true', help='Run in automatic continuous mode')
    args = parser.parse_args()

    client = mqtt.Client()
    client.connect(args.broker, 1883, 60)

    if args.auto:
        run_auto_mode(client)
        return

    # subscribe to control topic and wait for 'record' messages
    client.on_message = on_control_message
    client.subscribe(TOPIC_CONTROL)
    print("Listening for control messages on {} (broker={})".format(TOPIC_CONTROL, args.broker))
    client.loop_forever()


if __name__ == '__main__':
    main()