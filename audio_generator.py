import soundfile as sf
from something import Kokoro



def savenow(text):
    kokoro = Kokoro("kokoro-v1.0.onnx","voices-v1.0.bin")
    samples, sample_rate = kokoro.create(
        text, voice="af_bella", speed=1.0, lang="en-us"
)
    sf.write(file='1.wav',data=samples, samplerate=sample_rate)


savenow("How do you think I'll sound?")