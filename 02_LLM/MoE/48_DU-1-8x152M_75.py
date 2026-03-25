import os
import sys
import time
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# =====================================
# 設定
# =====================================
output_dir = "./finetuned_DU-1-8x152M_75"
device = "cuda" if torch.cuda.is_available() else "cpu"
transformers.utils.logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

# =====================================
# モデル・トークナイザ
# =====================================
tokenizer = AutoTokenizer.from_pretrained(
    output_dir,
    use_fast=False,
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.float32,
    use_safetensors=True,
    device_map="cuda"
)

model.eval()

# =====================================
# テストデータ
# =====================================
test_df = pd.read_csv("test_dialect.csv")
results = []

# =====================================
# 推論ループ
# =====================================
for i, row in test_df.iterrows():
    dialect = str(row["dialect"]).strip()
    prompt = f"方言: {dialect}\n標準語:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    standard = generated.split("標準語:")[-1].strip()
    standard = standard.split("方言:")[0].strip()

    results.append({
        "dialect": dialect,
        "generated_standard": standard,
        "expected_standard": str(row["standard"]).strip(),
        "latency_ms": round(elapsed_ms, 2)
    })

    print(f"--- {i+1}/{len(test_df)} ---")
    print(f"方言：{dialect}")
    print(f"標準語（生成）：{standard}")
    print(f"標準語（期待）：{row['standard']}")
    print(f"推論時間 : {elapsed_ms:.2f} [ms]")
    print()

# =====================================
# 保存
# =====================================
filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
timestamp = datetime.now().strftime("%H%M%S")
output_file = os.path.join("results", f"{filename}_{timestamp}.csv")

pd.DataFrame(results).to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("------------------------------------")
print(f"Finished testing {filename} !!")
print("------------------------------------")

# =====================================
# メモリ後処理
# =====================================
import gc

del model
del tokenizer
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
