#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install evaluate')


# In[3]:


get_ipython().run_line_magic('pip', 'install jiwer')


# In[ ]:


# --- RUN THIS CELL TO FIX THE ERROR ---

# 1. Force-uninstall any broken versions
print("Uninstalling old versions...")
get_ipython().system('pip uninstall -y torch torchaudio torchdata torchvision fastai')

# 2. Reinstall the correct, matching versions
print("\nInstalling fresh, matching versions...")
get_ipython().system('pip install torch torchaudio torchvision fastai torchcodec')

# 3. Restart runtime to apply changes
print("\nRestarting runtime...")
import os
os.kill(os.getpid(), 9)


# In[5]:


# ============================================================
# 🔊 Voice-Controlled Drone — S2T Model Fine-Tuning
# Author: M.raj Kumar
# Goal: Fine-tune the S2T-small model on the custom
#       drone command dataset. This model is RAM-friendly.
# VERSION 2.0 - Robust Pathing and MP3/Float Error Fix
# ============================================================

import os
import re
import zipfile
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import gc
import dataclasses
import string # Import the string module

# --- Core Hugging Face Libraries ---
import datasets
from datasets import load_dataset, Audio, DatasetDict
import transformers
from transformers import (
    Speech2TextForConditionalGeneration,
    Speech2TextProcessor,
    Trainer,
    TrainingArguments
)
import evaluate

# -------------------------------
# ✅ 1. CONFIGURATION
# -------------------------------
print("--- Configuration ---")
# --- Data Settings ---
# These are your paths in Colab
AUDIO_ZIP_PATH = "/content/Audios.zip"
METADATA_CSV_PATH = "/content/VCD-G2.csv"

# --- Internal Paths ---
TEMP_EXTRACT_DIR = "/content/temp_audio_extracted"
CLEANED_METADATA = "drone_dataset_for_training.csv"
# AUDIO_DIR = None # This will be found automatically - Removed

# --- Model Settings ---
BASE_MODEL_ID = "facebook/s2t-medium-librispeech-asr" # Set to medium model
FINETUNED_MODEL_PATH = "./drone_s2t_st_model"

# --- Training Settings ---
NUM_EPOCHS = 5 # S2T can fine-tune very quickly
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

# -------------------------------
# ✅ 2. EXTRACT AUDIO & PREPARE DATA
# -------------------------------
print(f"\n--- Loading Metadata ---")
try:
    df = pd.read_csv(METADATA_CSV_PATH)
except FileNotFoundError:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL ERROR: CSV File not found at {METADATA_CSV_PATH}")
    print(f"Please upload your 'VCD-G2.csv' file to Colab.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise

print(f"Found {len(df)} entries in {METADATA_CSV_PATH}")
if df.empty:
    raise ValueError("CSV file is empty.")

# Display DataFrame head and columns for debugging
print("\n--- Debugging Metadata ---")
print("DataFrame Head:")
display(df.head())
print("\nDataFrame Columns:")
print(df.columns)
print("-------------------------")


print(f"\n--- Extracting Audio ---")
try:
    with zipfile.ZipFile(AUDIO_ZIP_PATH, "r") as zip_ref:
        print(f"Extracting '{AUDIO_ZIP_PATH}' to '{TEMP_EXTRACT_DIR}'...")
        zip_ref.extractall(TEMP_EXTRACT_DIR)
    print("✅ Audio extraction complete.")

except FileNotFoundError:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL ERROR: ZIP File not found at {AUDIO_ZIP_PATH}")
    print(f"Please upload your 'Audios.zip' file to Colab.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise


print(f"\n--- Verifying Audio Files & Preparing Data ---")
def clean_text(text):
    # Convert to string, handle potential NaNs
    text = str(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower().strip()
    return text

prepared_data = []
# Iterate through each entry in the dataframe
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Verifying files"):
    file_name = row['LINK_1']
    transcript = row['TRANSCRIPT']

    # Check if file_name is a valid string and not NaN
    if isinstance(file_name, str) and not pd.isna(file_name):
        found_file_path = None
        # Search for the file in the entire extracted directory structure
        for root, dirs, files in os.walk(TEMP_EXTRACT_DIR):
            if file_name in files:
                found_file_path = os.path.join(root, file_name)
                break # Stop searching once the file is found

        if found_file_path:
            prepared_data.append({
                "file_path": found_file_path,
                "transcription": clean_text(transcript) # Apply the updated cleaning
            })
        else:
            print(f"Skipping row {index}: Audio file '{file_name}' not found in extracted directory.")
    else:
        print(f"Skipping row {index} due to invalid file name: {file_name}")


print(f"✅ Found {len(prepared_data)} matching audio/text pairs.")

if not prepared_data:
    raise ValueError("No valid audio files were found matching the entries in the CSV. Please check your CSV and zip file contents/structure.")

clean_df = pd.DataFrame(prepared_data)
clean_df.to_csv(CLEANED_METADATA, index=False)
print(f"✅ Cleaned metadata saved to: {CLEANED_METADATA}")

# -------------------------------
# ✅ 3. LOAD DATASET
# -------------------------------
print(f"\n--- Loading Dataset into Hugging Face ---")
dataset = load_dataset("csv", data_files={"train": CLEANED_METADATA})["train"]

# ------------------------------------------------------------------
# V V V V V V V V V V V  THIS IS THE FIX  V V V V V V V V V V V
# ------------------------------------------------------------------
# This ONE LINE tells `datasets` to load, decode, and resample
# the MP3s to 16kHz for us. This fixes all "float" errors.
print("Casting audio column (this will load/decode/resample all audio)...")
dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
# ------------------------------------------------------------------

# Split 90% train, 10% test
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(dataset)
print(f"✅ Dataset loaded. Training samples: {len(dataset['train'])}")

# -------------------------------
# ✅ 4. LOAD PROCESSOR & MODEL
# -------------------------------
print(f"\n--- Loading Model & Processor ---")
processor = Speech2TextProcessor.from_pretrained(BASE_MODEL_ID)
model = Speech2TextForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

model.config.use_cache = False
print(f"✅ Base model {BASE_MODEL_ID} loaded.")

# -------------------------------
# ✅ 5. PRE-PROCESS DATASET (FIXED)
# -------------------------------
print(f"\n--- Pre-processing Dataset (On-the-fly) ---")

# This function is now much simpler and safer
def prepare_dataset(batch):
    # `datasets` has already loaded and resampled the audio for us
    audio = batch["file_path"] # This is now an audio object, not just a path

    # The processor gets the correct float array and sample rate
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Tokenize transcript
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


print("Creating processing map...")
processed_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset['train'].column_names, # Remove all old columns
    num_proc=1 # Single-threaded for safety
)
print(f"✅ Dataset processing map created.")
# del dataset # Free up memory
gc.collect()

# -------------------------------
# ✅ 6. SET UP TRAINER
# -------------------------------
print(f"\n--- Configuring Trainer ---")

# Data collator pads the batches dynamically
@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: transformers.Speech2TextProcessor
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask != 1, -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluation metric
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # Use the S2T processor's tokenizer to decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training Arguments
training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps", # Changed from evaluation_strategy
    num_train_epochs=NUM_EPOCHS,
    fp16=True, # Use half-precision
    gradient_checkpointing=True, # Saves VRAM
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)
print(f"✅ Trainer initialized. Ready to start.")

# -------------------------------
# ✅ 7. RUN TRAINING
# -------------------------------
print(f"\n--- Starting Model Training ---")
print(f"Training for {NUM_EPOCHS} epochs...")
trainer.train()
print(f"✅ Training complete.")

# -------------------------------
# ✅ 8. SAVE FINAL MODEL
# -------------------------------
print(f"\n--- Saving Final Model ---")
trainer.save_model(FINETUNED_MODEL_PATH)
processor.save_pretrained(FINETUNED_MODEL_PATH)
print(f"✅ Fine-tuned model and processor saved to: {FINETUNED_MODEL_PATH}")

# -------------------------------
# ✅ 9. SHOW RESULTS (as requested)
# -------------------------------
print(f"\n--- Running Example Inference to Show Results ---")
print("Loading our new fine-tuned model...")
# Load the model and processor we just saved
model = Speech2TextForConditionalGeneration.from_pretrained(FINETUNED_MODEL_PATH)
processor = Speech2TextProcessor.from_pretrained(FINETUNED_MODEL_PATH)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Getting a random sample from the test set...")
# Get a random sample from the test set
# We need to get the "raw" test set sample *before* processing
# Let's get it from the original split dataset
raw_test_sample = dataset["test"][0]
audio_data = raw_test_sample["file_path"] # This is the audio object
true_label = raw_test_sample["transcription"]

# Manually process this one sample for inference
input_features = processor(
    audio_data["array"],
    sampling_rate=audio_data["sampling_rate"],
    return_tensors="pt"
).input_features.to(device)


# Generate prediction
print("Generating prediction...")
with torch.no_grad():
    generated_ids = model.generate(inputs=input_features, max_length=150)

# Decode the prediction and the true label
prediction = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
label = true_label # We already have the clean string

print("\n--- ✅ FINAL TEST RESULT ---")
print(f"Ground Truth:    {label.lower()}")
print(f"Prediction:      {prediction.lower()}")
print("----------------------------")

