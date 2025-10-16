import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np
fs = 16000
seconds = 5
model = whisper.load_model("tiny")
print("recording...")

try:
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
except KeyboardInterrupt:
    print("\nstopped")

audio = recording.flatten()
result = model.transcribe(
    audio, 
    fp16=False,
    language='en',
    beam_size=2,
    best_of=2,
    temperature=0.5
)
print(result["text"])


write("audio.wav", fs, (recording * 32767).astype('int16'))
print("audio.wav")
