import os

# set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNING'] = '1'


# Step 1: Whisper
from transformers import pipeline

asr = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-medium",
    device=0
)

raw_result = asr("dialect.wav", generate_kwargs={"task": "transcribe", "language": "ja"})
dialect_text = raw_result["text"]
print("方言:", dialect_text)

# Step 2: LLMで
from openai import OpenAI
client = OpenAI(api_key="...")

prompt = f"次の方言を自然な標準語に変換してください。\n\n{dialect_text}"

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # あるいは日本語LLM
    messages=[{"role": "user", "content": prompt}]
)

standard_text = resp.choices[0].message.content
print("標準語:", standard_text)
