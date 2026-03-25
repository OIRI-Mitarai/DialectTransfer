import os
import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, DataCollatorForSeq2Seq, WhisperForConditionalGeneration
from torch.utils.data import DataLoader

# モデルとプロセッサのパス
MODEL_PATH = "./whisper-medium"
DATASET_PATH = "./dialect-preprocessed"

try:
    # 1. 前処理済みデータセットを読み込む
    dataset = load_from_disk(DATASET_PATH)

    # 2. データセットの形式をPyTorchに設定
    dataset.set_format(type="torch", columns=["input_features", "input_ids"])

    # 3. モデルとプロセッサをローカルから読み込む
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="ja", task="transcribe")

    # 4. Data Collatorの準備
    data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, model=model, padding="longest")

    # 5. DataLoaderの作成
    # トレーニングのバッチサイズと一致させる
    dataloader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)

    print("DataLoader has been successfully created. Starting to iterate over batches...")

    # 6. バッチをイテレートし、中身を確認
    for i, batch in enumerate(dataloader):
        print(f"\n--- Batch {i+1} ---")
        for key, value in batch.items():
            print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")

        # 最初のバッチを確認できたら終了
        if i == 0:
            break

    print("\nBatching successful! The issue might be related to the Trainer's training loop.")

except Exception as e:
    print(f"\nAn error occurred during batching: {e}")
