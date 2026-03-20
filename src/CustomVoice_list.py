import json
import os
import sys
from pathlib import Path

# Ensure the project root is on `sys.path` so `import qwen_tts` works when running:
#   python src/CustomVoice.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
            instruct=(
                "請以標準普通話發音朗讀這個單字，發音必須：1) 單音節，不重複；2) 完全中性平直，無任何情緒或語氣；"
                "3) 起音乾淨清晰，無氣聲、嘆氣或吸氣聲；4) 結尾立即停止，不拖長、不帶氣尾；5) 音量穩定一致。"
                "這是用於聽能復健訓練的標準發音示範。"
            ),
            language="Chinese",
            speaker="Vivian",
            # 聽能復健專用：最大化一致性，最小化語氣變化
            do_sample=False,           # 確定性輸出
            temperature=0.005,          # 極低溫度，減少隨機性
            top_p=0.8,                 # 限制採樣範圍
            max_new_tokens=128,        # 限制長度，避免拖長
            repetition_penalty=1.2,    # 避免重複發音
        )
        sf.write(output_path, wavs[0], sr)
        print(f"  Saved: {output_path}")

print("Done.")
