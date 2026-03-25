import os
from transformers import AutoModelForCausalLM

BASE_DIR = "."  # 必要なら変更（カレントでOK）

for name in os.listdir(BASE_DIR):
    if not name.startswith("finetuned"):
        continue

    model_dir = os.path.join(BASE_DIR, name)

    if not os.path.isdir(model_dir):
        continue

    safetensor_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if not os.path.exists(safetensor_path):
        print(f"[SKIP] {name}: model.safetensors not found")
        continue

    if os.path.exists(bin_path):
        print(f"[SKIP] {name}: pytorch_model.bin already exists")
        continue

    print(f"[CONVERT] {name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )

    model.save_pretrained(
        model_dir,
        safe_serialization=False  # ← bin を生成
    )

    print(f"[DONE] {name}: pytorch_model.bin created")

print("All done.")
