from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model
from collections import defaultdict
import os
import json

# ============================================
# パラメータ
# ============================================
# model_name = "llm-jp/DU-1.0-8x152M"   ← ここだけ差し替えればOK
model_name = "llm-jp/DU-0.5-8x152M"
output_dir = "./finetuned_DU-0.5-8x152M_25"
num_train_epochs = 25


# ============================================
# モデルとトークナイザ（MoE対応）
# ============================================
print(f"[INFO] Loading model: {model_name}")

# use_fast=False は MoE 系で非推奨
tokenizer = AutoTokenizer.from_pretrained(model_name)

# pad_token が無い場合は eos を使う（MoE系で安全）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


# ============================================
# 方言辞書 Word Embedding（MoE 安全版）
# ============================================
DIALECT_DICT_PATH = "dialect_dict.json"
if os.path.exists(DIALECT_DICT_PATH):
    print(f"[INFO] Loading dialect dictionary from {DIALECT_DICT_PATH}")
    with open(DIALECT_DICT_PATH, "r", encoding="utf-8") as f:
        dialect_dict = json.load(f)

    # 方言語を特殊トークンとして追加
    special_tokens = [d["方言"] for d in dialect_dict]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # embedding 層をリサイズ
    model.resize_token_embeddings(len(tokenizer))

    # === MoE 安全版：標準語の1語をベクトルコピーする ===
    with torch.no_grad():
        emb = model.get_input_embeddings().weight

        for d in dialect_dict:
            d_tok = d["方言"]
            s_tok = d["標準語"]

            d_id = tokenizer.convert_tokens_to_ids(d_tok)
            s_id = tokenizer.convert_tokens_to_ids(s_tok)

            if d_id != tokenizer.unk_token_id and s_id != tokenizer.unk_token_id:
                # MoE では meanベクトルは危険 → コピー方式が最も安全
                emb[d_id] = emb[s_id]

    print(f"[INFO] Dialect embeddings initialized safely for MoE ({len(dialect_dict)} entries).")

else:
    print("[WARN] No dialect_dict.json found. Proceeding without dictionary embedding.")


# ============================================
# LoRA（MoE ではそのまま使えないのでコメントアウト）
# ============================================
"""
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["ATTENTION_Q", "ATTENTION_K"],  # MoE 用に自動検出する必要がある
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# model = get_peft_model(model, lora_config)
"""


# ============================================
# データセット
# ============================================
dataset = load_dataset("json", data_files="dialect2standard.jsonl", split="train")


# ============================================
# tokenize function
# ============================================
def tokenize(example):

    prompt = f"方言: {example['方言']}\n標準語:"
    answer = example["標準語"]

    # 入力＋出力を手動で連結
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]

    # 128トークンで切る
    if len(input_ids) > 128:
        input_ids = input_ids[:128]
        labels = labels[:128]

    pad_len = 128 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [-100] * pad_len
    attention_mask = [1] * (128 - pad_len) + [0] * pad_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


tokenized_dataset = dataset.map(tokenize, batched=False)


# ============================================
# 学習設定
# ============================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    save_steps=200,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
)


# ============================================
# Trainer
# ============================================
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


# ============================================
# テストフェーズ（MoEモデルは pipeline 非対応）
# ============================================
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = output_dir

tokenizer = AutoTokenizer.from_pretrained(model_dir)
generator = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=tokenizer,
    device=0
)

test_df = pd.read_csv("test_dialect.csv")
results = []

for i, row in test_df.iterrows():
    dialect = str(row["dialect"]).strip()
    prompt = f"方言: {dialect}\n標準語:"

    output = generator(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=True,
        output_router_logits=False
    )

    gen = output[0]["generated_text"]
    standard = gen.split("標準語:")[-1].strip()
    standard = standard.split("方言:")[0].strip()

    results.append({
        "dialect": dialect,
        "generated_standard": standard,
        "expected_standard": row["standard"]
    })

    print(f"--- {i+1}/{len(test_df)} ---")
    print(f"方言：{dialect}")
    print(f"標準語（生成）：{standard}")
    print(f"標準語（期待）：{row['standard']}")
    print()


# 結果保存
filename = os.path.splitext(os.path.basename(__file__))[0]
output_file = os.path.join("results", filename + ".csv")

pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")

print("------------------------------------")
print(f"Finished testing {filename} !!")
print("------------------------------------")
