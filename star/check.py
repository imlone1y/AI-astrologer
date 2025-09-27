import sys
import ollama
import numpy as np

def is_valid_answer(question: str, user_input: str) -> bool:
    prompt = f"""
你是一位負責檢查回答是否合理的助手。請你根據常識與語意，判斷下面的回答是否有回答到問題，並用一個詞回答「有」或「沒有」。
問題：
{question}

回答內容：
{user_input}

請你只回答「有」或「沒有」兩個字。
"""

    response = ollama.chat(
        model='gemma3:12b',
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response['message']['content'].strip()
    return reply.strip() == "有"

def is_valid_name(user_input: str):
    prompt = f"""
你是一位檢查助手，請你根據常識與語意，請從句子中提取使用者的姓名。
如果句子太多雜音或是不合理或無法提取，就回答”不清楚“。
whats your name 用戶的回答：
{user_input}

請你只回答解析出來的用戶名稱。    
"""
    response = ollama.chat(
        model='gemma3:12b',
        messages=[{"role": "user", "content": prompt}]
    )
    reply = response['message']['content'].strip()
    return reply


def is_silent(audio_np, threshold=0.01):
    # 均方根能量法判斷有沒有聲音（非完全靜音也能偵測）
    audio_float = audio_np.astype(np.float32)
    rms = np.sqrt(np.mean(audio_float ** 2))
    return rms < threshold


def normalize(date_str: str, time_str: str):
    prompt = f"""
你是一位時間格式轉換助手，請將以下使用者輸入的日期與時間轉為標準格式：
- 日期輸入：{date_str}
- 時間輸入：{time_str}

請輸出格式如下：
日期: YYYY-MM-DD
時間: HH:MM

只回答這兩行格式，不要多餘文字。
"""
    response = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    reply = response['message']['content'].strip()
    
    # 基礎解析
    date_out, time_out = None, None
    for line in reply.splitlines():
        if line.lower().startswith("日期:"):
            date_out = line.split(":", 1)[1].strip()
        elif line.lower().startswith("時間:"):
            time_out = line.split(":", 1)[1].strip()
    
    return date_out, time_out





# ✅ 測試模式用：直接從命令列輸入問題 + 回答
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：python AI_check.py '問題' '回答'")
        print("範例：python AI_check.py '你的生日是？' '13月32日'")
        sys.exit(1)

    question = sys.argv[1]
    answer = sys.argv[2]
    result = is_valid_answer(question, answer)

    print("✅ 合理的回答" if result else "❌ 不合理的回答")
