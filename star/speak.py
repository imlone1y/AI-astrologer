import pyttsx3

def speak(text):
    print(text)
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()  # ✅ 等待說完
    return True  # 說完後才回傳


