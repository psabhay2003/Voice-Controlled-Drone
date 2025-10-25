import torch
import sounddevice as sd

import wavio
import whisper
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_dir = "./final_model" 
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()


whisper_model = whisper.load_model("small")  


def record_audio(duration=5, fs=16000, filename="voice.wav"):

    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print("✅ Recording saved as", filename)
    return filename

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    text = result["text"]
    print("📝 Transcribed Text:", text)
    return text

def extract_distance(text):
    match = re.search(r"(\d+)\s*(meters|meter|m|feet|foot)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def predict_command(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return model.config.id2label[pred_id]



mic_device = None  # set to your device ID if needed (from sd.query_devices())

print("Voice Command System Started. Say 'stop' to exit.")

while True:
    try:
        audio_file = record_audio(duration=10)
        text = transcribe_audio(audio_file)
        print(" You said ->", text)
        
        if "stop" in text.lower():  # stop the loop
            print(" Stopping voice command system.")
            break

        distance = extract_distance(text)
        command = predict_command(text)

        print(f" Predicted Command: {command}")
        print(f" Predicted Distance: {distance} meters")

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print("⚠️ Error:", e)
        continue