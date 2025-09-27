import os
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from typing import Optional
from speechbrain.inference.speaker import SpeakerRecognition


# 初始化 ECAPA 模型（自動下載）
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def record_audio(duration_sec=2, sample_rate=16000) -> str:
    print(f"🎙️ 正在錄音 {duration_sec} 秒...")
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    # 儲存為臨時 wav 檔
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp_wav.name, sample_rate, audio)
    print(f"✅ 錄音完成，檔案：{tmp_wav.name}")
    return tmp_wav.name

def match_speaker(test_audio_path: str, profile_dir: str = "./profiles", threshold: float = 0.5) -> Optional[str]:
    best_match = (None, -1.0)

    for fname in os.listdir(profile_dir):
        if not fname.endswith(".wav"):
            continue
        profile_path = os.path.join(profile_dir, fname)

        try:
            score, prediction = spkrec.verify_files(profile_path, test_audio_path)
            print(f"📣 相似度 vs {fname}: {float(score):.4f}")
            if score > best_match[1]:
                best_match = (os.path.splitext(fname)[0], score)
        except Exception as e:
            print(f"⚠️ 無法比對 {fname}：{e}")
            continue

    if best_match[1] >= threshold:
        return best_match[0]
    return None

if __name__ == "__main__":
    test_audio_path = record_audio(duration_sec=2)

    name = match_speaker(test_audio_path)
    print("✅ 找到匹配者：" + name if name else "❌ 無符合使用者")

    # 刪除錄音暫存檔
    os.remove(test_audio_path)
