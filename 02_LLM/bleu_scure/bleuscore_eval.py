import os
import glob
import pandas as pd
from sacrebleu.metrics import BLEU

# ==========================
# 設定
# ==========================
RESULTS_DIR = "results"
OUTPUT_CSV = "bleu_summary.csv"

# ==========================
# BLEU 初期化
# ==========================
bleu = BLEU(tokenize="none")

def char_tokenize(sentences):
    """文字単位BLEU用の分かち書き"""
    return [" ".join(list(str(s))) for s in sentences]

# ==========================
# CSV を順番に処理
# ==========================
rows = []

csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))

if not csv_files:
    raise RuntimeError("results/ 配下にCSVが見つかりません")

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path, header=None)

        hypotheses = df[1].astype(str).tolist()
        references = df[2].astype(str).tolist()

        hypotheses_char = char_tokenize(hypotheses)
        references_char = [char_tokenize(references)]

        score = bleu.corpus_score(hypotheses_char, references_char)

        rows.append({
            "file_name": os.path.basename(csv_path),
            "bleu_score": round(score.score, 2)
            # "num_samples": len(df)
        })

        print(f"[OK] {os.path.basename(csv_path)} : BLEU = {score.score:.2f}")

    except Exception as e:
        print(f"[NG] {csv_path} : {e}")

# ==========================
# 結果保存
# ==========================
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("----------------------------------")
print("BLEU summary saved to:", OUTPUT_CSV)
print("----------------------------------")
