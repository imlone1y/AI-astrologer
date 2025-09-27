# AI 占星機器人

繁體中文 | [English](README.md)

本項目處於開發階段，未經授權禁止使用、販售。

## 項目介紹

本項目為設計一款 AI 占星終端產品，此款產品具有喚醒、聲紋辨識、占星、用戶星盤計算等功能，程式碼分為伺服器端與客戶端，此處僅展示單機版。本項目在 `macOS 15.6.1` 上，以 `python 3.13.3` 版本完成測試。

## 項目結構
```
.
├── astrolabe.py                      # 計算用戶星盤
├── check.py                          # 檢查用戶回答是否合法
├── files                             # 檢索檔案資料夾
│   └── 518091629-鲁道夫-占星全书.pdf   # 占星書籍
├── hey_astrovia.onnx                 # 使用 OpenWakeWord 訓練出喚醒詞檔案
├── hey_astrovia.tflite               # 使用 OpenWakeWord 訓練出喚醒詞檔案
├── main.py                           # 主程式
├── pretrained_models                 # 聲紋辨識模型
│   ├── skip..
├── profiles                          # 使用者資料資料夾
│   ├── justin.txt                    # 使用者資料與星盤
│   └── justin.wav                    # 使用者聲音，用於聲紋辨識
├── rag.py                            # 強化檢索生成
├── requirements.txt         
└── voice_matcher.py                  # 聲紋辨識
```

## 項目使用套件

- 喚醒功能：[openWakeWord](https://github.com/dscripka/openWakeWord)
- STT: python `faster-whisper`
- TTS: [coqui TTS](https://github.com/coqui-ai/TTS)，框架使用 [alltalk tts](https://github.com/erew123/alltalk_tts)
- RAG: [bge-m3](https://huggingface.co/BAAI/bge-m3)
- LLM: [gemma3:12b](https://ollama.com/library/gemma3:12b)
- 聲紋辨識：[spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## 項目流程圖
