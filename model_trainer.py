import pandas as pd
import openpyxl

d1=pd.read_csv('/VCD_DATA.csv')
d2=pd.read_csv('/dataset_links.csv')
d1['drive_link']=d2['drive_link']
d1.to_csv('/VCD_DATA.csv', index=False)


import os
import re
import gdown
import pandas as pd


csv_file = "/dataset.csv"
output_folder = "audio"

os.makedirs(output_folder, exist_ok=True)


df = pd.read_csv(csv_file)

def extract_file_id(link):
    if not isinstance(link, str):
        return None
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    return match.group(1) if match else None

skipped=0
invalid=0
for i, row in df.iterrows():
    name = str(row.get("RECORDING FILE", f"audio_{i+1}"))
    link = row.get("drive_link")

    if not link:
        print(f"Skipping {name}: No link")
        continue

    file_id = extract_file_id(link)
    if not file_id:

        print(f"Skipping {name}: Invalid link")
        invalid+=1
        print(invalid)
        continue
    print("File Id",file_id)
    url = f"https://drive.google.com/uc?id={file_id}"
    print("url",{url})
    if not name.lower().endswith((".mp3", ".wav", ".m4a")):
        name += ".mp3"

    output_path = os.path.join(output_folder, name)

    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed: {name} -> {e}")
        skipped+=1
        print("skipped:",skipped)
        print(f"Skipping this file and continuing...\n")
        continue

import os
import pandas as pd
import whisper

# Inputs
input_csv = "/dataset.csv"
audio_folder = "/content/voice"
output_csv = "transcripts.csv"


df = pd.read_csv(input_csv)
model = whisper.load_model("base")


transcript_cache = {}

# for each row
for i, row in df.iterrows():
    audio_name = str(row["RECORDING FILE"]).strip()
    file_path = None

    for ext in [".mp3", ".wav", ".m4a"]:
        test_path = os.path.join(audio_folder, audio_name)
        if os.path.exists(test_path):
            file_path = test_path
            break
    if not file_path:
        print(f" Missing audio file for: {audio_name}")
        df.at[i, "whisper_text"] = ""
        continue

    if file_path in transcript_cache:
        text = transcript_cache[file_path]
    else:
        print(f"🎧 Transcribing: {audio_name}")
        result = model.transcribe(file_path)
        text = result["text"].strip()
        transcript_cache[file_path] = text

    df.at[i, "whisper_text"] = text

# save
df.to_csv(output_csv, index=False)
print(f"\n Transcription completed and saved to: {output_csv}")


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv("/content/dataset_with_transcripts.csv")

# Use only valid rows
df = df[["whisper_text", "COMMAND"]].dropna()

# Encode labels
unique_labels = sorted(df["COMMAND"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["COMMAND"].map(label2id)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Load tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["whisper_text"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")

print("Model and tokenizer saved to ./results/final_model")

import numpy as np

# Put model in evaluation mode
model.eval()

def predict_command(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return id2label[pred_id]

# Apply prediction to all whisper_texts
df["predicted_command"] = df["whisper_text"].apply(predict_command)

# Save updated dataset
output_path = "/content/dataset_with_predicted_commands.csv"
df.to_csv(output_path, index=False)

print(f"✅ Predictions saved to {output_path}")



import re

def extract_distance(text):

    if not isinstance(text, str):
        return None

    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(meters|meter|m)\b", text.lower())
    if match:
        return float(match.group(1))  # return as number
    return None

# Apply it
df["predicted_distance"] = df["whisper_text"].apply(extract_distance)

df.to_csv("/content/dataset_with_predicted_commands_and_distance.csv", index=False)
print("Saved with predicted commands and distances.")



import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

test_path = "/content/test_commands.csv"
df_test = pd.read_csv(test_path)


model_dir = "./results/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)


model.eval()


def predict_command(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return model.config.id2label[(pred_id)]

df_test["predicted_command"] = df_test["TEXT"].apply(predict_command)

accuracy = accuracy_score(df_test["COMMAND"], df_test["predicted_command"])
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(df_test["COMMAND"], df_test["predicted_command"]))

df_test.to_csv("/content/test_predictions.csv", index=False)
print("📁 Results saved to /content/test_predictions.csv")
