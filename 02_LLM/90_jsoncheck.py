import json

with open("dialect2standard.jsonl", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        try:
            obj = json.loads(line)
            if not isinstance(obj["標準語"], str):
                print(f"❌ row {idx}: 標準語が文字列じゃない: {obj['標準語']}")
        except Exception as e:
            print(f"❌ row {idx}: JSONとして壊れている: {e}")
