from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model
from collections import defaultdict
import os
import json

# パラメータ
model_name = "rinna/japanese-gpt-neox-3.6b"
output_dir="./finetuned_rinna-gpt-neox_75"
num_train_epochs=75

# モデルとトークナイザ
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Word Embedding
DIALECT_DICT_PATH = "dialect_dict.json"
if os.path.exists(DIALECT_DICT_PATH):
    print(f"[INFO] Loading dialect dictionary from {DIALECT_DICT_PATH}")
    with open(DIALECT_DICT_PATH, "r", encoding="utf-8") as f:
        dialect_dict = json.load(f)

    # 方言語を特殊トークンとして追加
    special_tokens = [d["方言"] for d in dialect_dict]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # モデルのembedding層をリサイズ
    model.resize_token_embeddings(len(tokenizer))

    # === 修正版：複数標準語の平均embeddingに更新 ===
    emb_map = defaultdict(list)
    for d in dialect_dict:
        emb_map[d["方言"]].append(d["標準語"])

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        for dialect, std_list in emb_map.items():
            d_id = tokenizer.convert_tokens_to_ids(dialect)
            std_ids = [tokenizer.convert_tokens_to_ids(s) for s in std_list if s != tokenizer.unk_token_id]
            if d_id != tokenizer.unk_token_id and std_ids:
                emb[d_id] = torch.stack([emb[sid] for sid in std_ids]).mean(dim=0)

    print(f"[INFO] Dialect embedding initialized for {len(dialect_dict)} entries (mean-based).")
else:
    print("[WARN] No dialect_dict.json found. Proceeding without dictionary embedding.")

# LoRA 設定
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,              # orig:32
    # target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # GPT-neox系
    target_modules=["query_key_value"],  # GPT-neox系
    lora_dropout=0.05,           # orig:0.05
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA適用
model = get_peft_model(model, lora_config)
# LoRAの層だけを学習対象
model.enable_input_require_grads()  # ← 勾配追跡ON（これが無いとNone grad）
model.gradient_checkpointing_enable()  # ← チェックポイント有効
model.config.use_cache = False


# 訓練対象パラメータの確認
trainable, total = 0, 0
for name, param in model.named_parameters():
    total += param.numel()
    if param.requires_grad:
        print(name)
        trainable += param.numel()
print(f"Trainable params: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.2f}%)")


# データセット
dataset = load_dataset("json", data_files="dialect2standard.jsonl", split="train")

# トークナイズ関数
def tokenize(example):
    # 入力（プロンプト）
    prompt = f"方言: {example['方言']}\n標準語:"
    # 出力
    answer = example["標準語"]

    # トークナイズ（入力と出力を連結）
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]

    # ラベルを作成（入力部分は -100 にして損失を無視）
    labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]

    # padding/truncation
    if len(input_ids) > 128:
        input_ids = input_ids[:128]
        labels = labels[:128]

    padding_len = 128 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_len
    labels += [-100] * padding_len
    attention_mask = [1] * (len(input_ids) - padding_len) + [0] * padding_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
'''
def tokenize(example):
    prompt = f"指示: {example['instruction']}\n方言: {example['dialect']}\n標準語:"
    model_inputs = tokenizer(
        prompt,
        text_target=example["standard"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    return model_inputs
'''

tokenized_dataset = dataset.map(tokenize, batched=False)

# 学習設定 本命
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    save_steps=200,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=20,
    fp16=True,
    gradient_checkpointing=True,
)

'''
# 学習設定 弱い(ほぼファインチューニングなし比較用)
training_args = TrainingArguments(
    output_dir="./finetuned-dialect2standard",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_steps=1,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
)
'''

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)

print('---------------------------')
print('LLM fine-tuning was DONE !!')
print('---------------------------')



# ここからコピペ
import os
import sys
import pandas as pd
from transformers import pipeline

# モデルとトークナイザのロード
model_dir = output_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir)

generator = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=tokenizer,
    device=0  # GPUを使う。CPUなら -1
)

# テストデータの読み込み
test_df = pd.read_csv("test_dialect.csv")

# 結果格納用リスト
results = []

# 各方言文に対して変換実行
for i, row in test_df.iterrows():
    dialect = str(row["dialect"]).strip()
    prompt = f"方言: {dialect}\n標準語:"

    # 生成
    output = generator(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=True
    )

    generated = output[0]["generated_text"]

    # 「標準語:」以降の部分を抽出
    standard = generated.split("標準語:")[-1].strip()
    standard = standard.split("方言:")[0].strip()
    standard = standard.split("標準語:")[0].strip()

    # 結果を保存
    results.append({
        "dialect": dialect,
        "generated_standard": standard,
        "expected_standard": str(row["standard"]).strip()
    })

    # 途中経過をターミナルにも表示
    print(f"--- {i+1}/{len(test_df)} ---")
    print(f"方言：{dialect}")
    print(f"標準語（生成）：{standard}")
    print(f"標準語（期待）：{row['standard']}")
    print()

# 出力先設定
filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
output_file = os.path.join("results", filename + ".csv")

# 結果をDataFrameにまとめて保存
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("------------------------------------")
print("Finished testing fine-tuned model !!")
print("------------------------------------")
