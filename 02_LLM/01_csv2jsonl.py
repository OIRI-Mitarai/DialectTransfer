import pandas as pd
import json
import csv
import os

'''
短文層向け：CSV → JSONL
'''
df = pd.read_csv("dialect2standard.csv")

with open("dialect2standard.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = {
            "instruction": "次の方言を標準語に変換してください。",
            "方言": row["dialect"],
            "標準語": row["standard"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("---------------------------------------")
print("FINISHED CREATE dialect2standard.jsonl")
print("---------------------------------------")


'''
辞書層向け：CSV → JSON
'''
data = []

with open("dialect_dict.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # キーの余分な空白や不可視文字を除去
        clean_row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        dialect = clean_row.get("dialect", "")
        standard = clean_row.get("standard", "")
        if dialect and standard:
            data.append({"方言": dialect, "標準語": standard})

with open("dialect_dict.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("---------------------------------")
print("FINISHED CREATE dialect_dict.json")
print("---------------------------------")
