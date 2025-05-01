import sounddevice as sd
from something import Kokoro



def playnow(text):
    kokoro = Kokoro("kokoro-v1.0.onnx","voices-v1.0.bin")
    samples, sample_rate = kokoro.create(
        text, voice="af_heart", speed=1.0, lang="en-us"
)
    sd.play(samples, sample_rate)