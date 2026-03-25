from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model

# モデルとトークナイザ
model_name = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA 設定
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,              # orig:32
    target_modules=["c_attn"],  # GPT2系
    lora_dropout=0.05,           # orig:0.05
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

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
    output_dir="./finetuned-dialect2standard",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=150,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    save_steps=200,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
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
trainer.save_model("./finetuned-dialect2standard")

print('---------------------------')
print('LLM fine-tuning was DONE !!')
print('---------------------------')
