import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODEL_PATH = "./whisper-medium"
DATASET_PATH = "./dialect-preprocessed"

# 1. Load datasets and model, processor
dataset = load_from_disk(DATASET_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="ja", task="transcribe")

# 2. Set the dataset format to PyTorch
dataset.set_format(type="torch", columns=["input_features", "labels"])

# 3. Specifically set generation settings for a model
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
# Disable gradient checkpoint
model.gradient_checkpointing_disable()

# 4. Train settings
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetune",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    # max_steps=20,
    num_train_epochs=50,
    save_steps=50,
    logging_steps=20,
    # evaluation_strategy="no",
    predict_with_generate=True,
    fp16=False,
    bf16=True,
)

# 5. Data collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Input audio
        input_features = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)

        # Labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace "attention_mask=0" with "-100", exclude from loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return {"input_features": input_features, "labels": labels}

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# 6. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)


# 7. Start training
trainer.train()

# 8. Save model in local
trainer.save_model("./whisper-finetune/final")
processor.save_pretrained("./whisper-finetune/final")


print('=====================================')
print('finetuning finished !!!')
print('=====================================')
