import json
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 載入模型
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 載入字詞表
with open("data/word_lists.json", "r", encoding="utf-8") as f:
    word_lists = json.load(f)

# 建立輸出資料夾
for list_name in word_lists:
    folder = os.path.join("result", list_name)
    os.makedirs(folder, exist_ok=True)

# 逐字產生音檔
for list_name, words in word_lists.items():
    folder = os.path.join("result", list_name)
    print(f"Processing {list_name} ({len(words)} words)...")

    for word in words:
        output_path = os.path.join(folder, f"{list_name}_{word}.wav")

        # 若已存在則跳過
        if os.path.exists(output_path):
            print(f"  Skip (exists): {output_path}")
            continue

        wavs, sr = model.generate_custom_voice(
            text=word,
            instruct="請用平穩、沒有起伏的機器人語氣唸出這個字",
            language="Chinese",
            speaker="Vivian",
        )
        sf.write(output_path, wavs[0], sr)
        print(f"  Saved: {output_path}")

print("Done.")
