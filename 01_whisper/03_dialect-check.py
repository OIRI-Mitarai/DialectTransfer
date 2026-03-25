from datasets import load_from_disk

try:
    # データセットを読み込む
    dataset = load_from_disk("./dialect-preprocessed")

    # データセットのスキーマ（列名と型）を表示
    print("--- Dataset Schema ---")
    print(dataset["train"].features)
    print("\n")

    # 最初のサンプルの中身を表示
    print("--- First Sample ---")
    first_sample = dataset["train"][0]
    for key, value in first_sample.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        if key == "input_ids":
            print(f"  Value: {value}")
        print("-" * 20)

except Exception as e:
    print(f"Error loading dataset: {e}")



print(dataset)
print(dataset["train"][0].keys())
