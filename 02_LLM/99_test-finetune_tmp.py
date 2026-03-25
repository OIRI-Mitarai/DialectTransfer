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
