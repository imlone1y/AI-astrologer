import pyaudio
import numpy as np
import soundfile as sf
import time
from faster_whisper import WhisperModel
from check import is_valid_answer, is_silent, is_valid_name
import os
from dotenv import load_dotenv
from voice_matcher import match_speaker
import noisereduce as nr
from openwakeword.model import Model
from openwakeword import utils
from speak import speak
from astrolabe import generate_chart_from_user_input
from rag import rag_ollama, embed_files
from typing import Tuple


# 下載預設 wakeword 模型
utils.download_models()

load_dotenv()

whisper_model = WhisperModel("large", compute_type="float32")

listener_stream = None
listener_audio = None

# 錄音參數
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

wakeword_model = Model(
    wakeword_models=["./hey_astrovia.onnx"],
    inference_framework="onnx"
)

THRESHOLD = 0.6
KEYWORD_NAME = os.getenv("KEYWORD_NAME") or "hey astrovia"

questions = [
    "What's your name?",
    "Next, what's your date of birth? For example, July 6th, 2004.",
    "And what time were you born? For example, 3 PM or 10:30 AM.",
    "Last but not least, where were you born? For example, Taipei City."
]


def record_until_silence(prompt_text: str = None, language: str = "en") -> Tuple[str, np.ndarray]:
    if prompt_text:
        print("🗣️ " + prompt_text)
        speak(prompt_text)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silence_threshold = 250
    silence_seconds = 1.5
    silence_limit = int(silence_seconds * RATE / CHUNK)
    silence_count = 0

    print("🎙️ Recording... (will stop after silence)")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        audio_np = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

        if rms < silence_threshold:
            silence_count += 1
        else:
            silence_count = 0

        if silence_count > silence_limit:
            print("⏹️ Silence detected. Ending recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    raw_audio = np.frombuffer(b''.join(frames), dtype=np.int16)

    if is_silent(raw_audio):
        speak("I couldn't hear you clearly. Could you please repeat?")
        return record_until_silence(prompt_text, language)

    # 處理音訊
    clean_audio = nr.reduce_noise(y=raw_audio, sr=RATE)
    clean_audio = clean_audio.astype(np.float32) / (np.max(np.abs(clean_audio)) + 1e-6)

    if np.isnan(clean_audio).any() or np.isinf(clean_audio).any():
        print("⚠️ Audio contains NaN or Inf.")
        return record_until_silence(prompt_text, language)

    transcript = transcribe_local(clean_audio, language=language)
    return transcript.strip(), clean_audio


def transcribe_local(audio_np, language="en"):
    segments, _ = whisper_model.transcribe(audio_np, language=language)
    for segment in segments:
        if segment.text.strip():
            return segment.text.strip()
    return ""

def listen_for_keyword(cooldown_until):
    """
    監聽喚醒詞。若仍在 cooldown 期間，直接睡 0.1 秒返回 False
    """
    if time.time() < cooldown_until:
        time.sleep(0.1)
        return False

    global listener_stream, listener_audio

    listener_audio = pyaudio.PyAudio()
    listener_stream = listener_audio.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=1600
    )

    consecutive_hits = 0
    REQUIRED_HITS = 2
    LOCAL_THRESHOLD = 0.80

    try:
        while True:
            pcm = listener_stream.read(1600, exception_on_overflow=False)
            scores = wakeword_model.predict(np.frombuffer(pcm, dtype=np.int16))
            score = scores.get("hey_astrovia", 0.0)

            if score > LOCAL_THRESHOLD:
                consecutive_hits += 1
                if consecutive_hits >= REQUIRED_HITS:
                    print(f"✅ Wake word detected! Score={score:.2f}")
                    return True
            else:
                consecutive_hits = 0
    finally:
        listener_stream.stop_stream()
        listener_stream.close()
        listener_audio.terminate()
        listener_stream = None
        listener_audio = None

def ask_and_identify():
    speak("Hi, I'm your astrologer. What would you like to ask me today?")
    
    # ✅ 只要這一行就完成錄音、降噪、轉文字
    question, question_np = record_until_silence()

    # ✅ 儲存臨時音檔用來聲紋辨識
    sf.write("user_question.wav", question_np, RATE)

    matched_name = match_speaker("user_question.wav")
    os.remove("user_question.wav")  # 清除臨時檔

    if matched_name:
        print(f"✅ Recognized user: {matched_name}")
        return question, matched_name
    else:
        print("❌ Unknown user. Start asking birth data...")
        speak("Seems like you're someone I haven't met before. Let me ask you a few questions before we begin.")
        time.sleep(0.5)
        return question, None

def continue_conversation(matched_name):
    """
    在回答完一題後，繼續等待使用者進一步提問，直到 5 秒內沒人講話才結束。
    """
    timeout_seconds = 5
    print("⏳ Waiting for follow-up questions...")

    while True:
        start_time = time.time()
        question, audio = record_until_silence(prompt_text=None)  # 沒講話會 return 空字串

        # 若整段都沒講話（靜音）
        if question.strip() == "":
            elapsed = time.time() - start_time
            if elapsed < timeout_seconds:
                print(f"🕒 Only {elapsed:.2f}s of silence, retrying...")
                continue
            else:
                print("⛔ No speech detected. Ending follow-up mode.")
                break

        print(f"🧠 Follow-up question: {question}")
        if "no" in question.lower():
            speak("Okay, bye! See you next time.")
            global cooldown_until
            cooldown_until = time.time() + 2
            break
        sf.write("followup.wav", audio, RATE)
        matched = match_speaker("followup.wav")
        os.remove("followup.wav")

        if matched == matched_name:
            print(f"✅ Same user: {matched}")
        else:
            print(f"🔄 Switching user: {matched or 'unknown'}")
            matched_name = matched

        if matched_name:
            rag_result = rag_ollama(question, matched_name)
            speak(rag_result)
        else:
            speak("I'm sorry, I couldn't recognize you. Please say 'hey astrovia' again to start over.")
            break


cooldown_until = 0
embed_files()

while True:
    print(f"listen for {KEYWORD_NAME}")
    if listen_for_keyword(cooldown_until):
        result = ask_and_identify()
        cooldown_until = time.time() + 4

        if result is not None:
            question, matched_name = result

            # ➤ 若無法辨識聲紋，開始問基本資料
            if matched_name is None:
                answers = []
                audio_segments = []

                for idx, question_text in enumerate(questions):
                    while True:
                        ans, audio = record_until_silence(question_text)
                        if idx == 0:
                            ans = is_valid_name(ans)
                            if ans == "不清楚":
                                speak("can you answer again? I can't hear you.")
                                continue
                        elif idx in (1, 2):
                            if not is_valid_answer(question_text, ans):
                                print(f"❌ Invalid answer: {ans}. Please try again.")
                                continue
                        answers.append(ans)
                        audio_segments.append(audio)
                        break

                # ➤ 儲存聲音檔
                combined_audio = np.concatenate(audio_segments)
                filename = answers[0] if answers[0] else "unknown"
                filename = filename.lower().replace(" ", "_")
                sf.write(f"./profiles/{filename}.wav", combined_audio, RATE)
                print(f"🎉 Saved audio to {filename}.wav")

                generate_chart_from_user_input(
                    name=answers[0],
                    date_str=answers[1],
                    time_str=answers[2],
                    city_name=answers[3],
                )

                matched_name = filename  # 用檔名當識別名

            # ➤ 回答原本的問題（無論是辨識還是剛輸入資料）
            print(f"🧠 Transcribed question: {question}")
            rag_result = str(rag_ollama(question, matched_name))
            speak(rag_result)
            
            continue_conversation(matched_name)

