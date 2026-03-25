import os
import torch
from datasets import load_dataset, Audio, load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# PATH setting(model, preprocessed-dataset)
MODEL_PATH = "./whisper-medium"
DATASET_PATH = "./dialect-preprocessed"

# 1. Prepare datasets
# Execute creation if preprocessed datasets don't exist
if not os.path.exists(DATASET_PATH):
    print("Preprocessed dataset not found. Starting preprocessing...")
    # Load data
    dataset = load_dataset("csv", data_files="dialect.csv")
    dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

    # Load model and processor, first loading is from online
    model_name = "openai/whisper-medium"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language="ja", task="transcribe")

    # Save model and processor in local
    model.save_pretrained(MODEL_PATH)
    processor.save_pretrained(MODEL_PATH)

    # preprocess data
    def preprocess(batch):
        audio = batch["file"]
        input_features = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        input_ids = processor.tokenizer(batch["text"]).input_ids

        # Specifically Convert input_ids to tensor
        return {"input_features": input_features, "labels": torch.tensor(input_ids, dtype=torch.long)}

    # map preocess
    # num_proc 10 //Core Ultra 9 285K=24cores
    dataset = dataset.map(preprocess, remove_columns=["file", "text"], num_proc=1)

    # Save preprocessed dataset in local
    dataset.save_to_disk(DATASET_PATH)

else:
    print("Preprocessed dataset found. Loading from disk...")
    # Load preprocessed dataset
    dataset = load_from_disk(DATASET_PATH)
    # Load mocel and processor from local
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="ja", task="transcribe")
