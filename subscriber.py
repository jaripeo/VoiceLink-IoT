import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.io.wavfile import write, read

BROKER = "172.20.10.4"   # same broker IP
TOPIC = "audio/sample"

def on_message(client, userdata, msg):
    print("Received WAV file!")

    # save to disk
    with open("received.wav", "wb") as f:
        f.write(msg.payload)

    # play audio
    sr, audio = read("received.wav")
    sd.play(audio, sr)
    sd.wait()
    print("Played audio.\n")

client = mqtt.Client()
client.on_message = on_message

client.connect(BROKER, 1883, 60)
client.subscribe(TOPIC)

print("Listening for incoming audio...")

client.loop_forever()
