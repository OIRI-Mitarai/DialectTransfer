# データのキー確認
print("Sample from dataset:", dataset["train"][0])

# collator動作確認
from torch.utils.data import DataLoader
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)

batch = next(iter(loader))
print("Batch keys after collator:", batch.keys())
print("input_features:", batch["input_features"].shape)
print("labels:", batch["labels"].shape)
