# ============================================
# テストフェーズ（そのまま）
# ============================================
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from collections import defaultdict
import os
import json


def test_func(output_dir, result):
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
    filename = os.path.splitext(os.path.basename(result))[0]
    output_file = os.path.join("results", filename + ".csv")

    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")

    print("------------------------------------")
    print(f"Finished testing {result} !!")
    print("------------------------------------")



test_func("finetuned_RNU-0.5-8x152M_10", "40_RNU-0.5-8x152M_10")
test_func("finetuned_NU-8x152M_10", "41_NU-8x152M_10")
test_func("finetuned_DU-1.0-8x152M_10", "42_DU-1-8x152M_10")
test_func("finetuned_DU-0.5-8x152M_10", "43_DU-0.5-8x152M_10")
test_func("finetuned_BTX-8x152M_10", "44_BTX-8x152M_10")
