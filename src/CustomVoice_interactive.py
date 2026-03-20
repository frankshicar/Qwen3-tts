# coding=utf-8
import os
import sys
from pathlib import Path

# Ensure the project root is on `sys.path` so `import qwen_tts` works when running:
#   python src/CustomVoice_interactive.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


INSTRUCT_NEUTRAL_SINGLE_CHAR = (
    "請以標準普通話發音朗讀這個單字，發音必須：1) 單音節，不重複；2) 完全中性平直，無任何情緒或語氣；"
    "3) 起音乾淨清晰，無氣聲、嘆氣或吸氣聲；4) 結尾立即停止，不拖長、不帶氣尾；5) 音量穩定一致。"
    "這是用於聽能復健訓練的標準發音示範。"
)


def main():
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    out_dir = os.path.join("result", "interactive")
    os.makedirs(out_dir, exist_ok=True)

    print("Ready. 請輸入單一中文字（輸入 q 退出）。")
    while True:
        raw = input("字：").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        if not raw:
            continue
        if len(raw) != 1:
            print("請輸入「單一」漢字（例如：先）。")
            continue

        char = raw
        output_path = os.path.join(out_dir, f"single_{char}.wav")

        wavs, sr = model.generate_custom_voice(
            text=char,
            instruct=INSTRUCT_NEUTRAL_SINGLE_CHAR,
            language="Chinese",
            speaker="Vivian",
            # 聽能復健專用：最大化一致性，最小化語氣變化
            do_sample=False,           # 確定性輸出
            temperature=0.01,          # 極低溫度，減少隨機性
            top_p=0.6,                 # 限制採樣範圍
            max_new_tokens=128,        # 限制長度，避免拖長
            repetition_penalty=1.2,    # 避免重複發音
        )
        sf.write(output_path, wavs[0], sr)
        print(f"Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()



def batch_regenerate():
    """批次重新生成所有音檔，使用優化後的參數"""
    import json
    
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    
    # 讀取字表
    word_lists_path = os.path.join(PROJECT_ROOT, "data", "word_lists.json")
    with open(word_lists_path, "r", encoding="utf-8") as f:
        word_lists = json.load(f)
    
    for list_name, chars in word_lists.items():
        out_dir = os.path.join("result", list_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n處理 {list_name} ({len(chars)} 個字)...")
        for char in chars:
            output_path = os.path.join(out_dir, f"{list_name}_{char}.wav")
            
            wavs, sr = model.generate_custom_voice(
                text=char,
                instruct=INSTRUCT_NEUTRAL_SINGLE_CHAR,
                language="Chinese",
                speaker="Vivian",
                do_sample=False,
                temperature=0.05,
                top_p=0.8,
                max_new_tokens=128,
                repetition_penalty=1.2,
            )
            sf.write(output_path, wavs[0], sr)
            print(f"  ✓ {char}")
    
    print("\n批次生成完成！")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_regenerate()
    else:
        main()
