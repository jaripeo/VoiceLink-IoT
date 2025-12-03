import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time

BROKER = "172.20.10.4"   # Broker IP
TOPIC = "audio/sample"

SAMPLE_RATE = 16000         # good for speech
DURATION = 3                # seconds per sample

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

def record_audio():
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()
    print("Recording finished.")
    return audio

def publish_wav(audio):
    # save temporary wav
    write("temp.wav", SAMPLE_RATE, audio)

    # read file as bytes
    with open("temp.wav", "rb") as f:
        data = f.read()

    # publish via MQTT
    client.publish(TOPIC, data)
    print("Published audio sample.\n")

while True:
    audio = record_audio()
    publish_wav(audio)
    time.sleep(1)   # 1 second pause between cycles
