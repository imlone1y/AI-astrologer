from flatlib.chart import Chart
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib import const
from geopy.geocoders import Nominatim
import re
from datetime import datetime
from check import normalize


# ===== 日期與時間解析 =====

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12"
}

def parse_birth_date(text):
    text = text.lower().replace(",", "").strip()
    year = None
    month = None
    day = None

    # 年份
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if year_match:
        year = year_match.group(0)

    # 月份 + 日期
    for mon in MONTH_MAP:
        if mon in text:
            month = MONTH_MAP[mon]
            day_match = re.search(rf"{mon} (\d{{1,2}})(st|nd|rd|th)?", text)
            if day_match:
                day = f"{int(day_match.group(1)):02d}"
            break

    # 日期在前：6th July
    if not day:
        day_match = re.search(r'(\d{1,2})(st|nd|rd|th)? (january|february|march|april|may|june|july|august|september|october|november|december)', text)
        if day_match:
            day = f"{int(day_match.group(1)):02d}"
            month = MONTH_MAP[day_match.group(3)]

    if year and month and day:
        return f"{year}/{month}/{day}"
    
    # ➕ LLM fallback
    print("⚠️ 使用者日期格式無法解析，嘗試使用 LLM 修正")
    fixed_date, _ = normalize(text, "")
    return fixed_date.replace("-", "/") if fixed_date else None


def parse_birth_time(text):
    text = text.strip().lower()
    for fmt in ["%I%p", "%I:%M%p", "%H:%M"]:
        try:
            return datetime.strptime(text, fmt).strftime("%H:%M")
        except:
            continue

    # ➕ LLM fallback
    print("⚠️ 使用者時間格式無法解析，嘗試使用 LLM 修正")
    _, fixed_time = normalize("", text)
    return fixed_time if fixed_time else None


# ===== 城市 → 經緯度 =====

def get_city_latlon(city_name):
    geolocator = Nominatim(user_agent="astrobot")
    location = geolocator.geocode(city_name, timeout=5)
    if location:
        lat = location.latitude
        lon = location.longitude

        # 轉 flatlib 格式
        lat_dir = 'n' if lat >= 0 else 's'
        lon_dir = 'e' if lon >= 0 else 'w'
        lat_deg = f"{abs(int(lat))}{lat_dir}{int((abs(lat) % 1) * 60):02d}"
        lon_deg = f"{abs(int(lon))}{lon_dir}{int((abs(lon) % 1) * 60):02d}"
        return lat_deg, lon_deg, location.address
    return None, None, None

# ===== 主程式 =====

def generate_chart_from_user_input(
    name, date_str, time_str, city_name,
    qa_pairs=None,
    timezone="+08:00"
):
    name = name.lower()
    output_path=f"./profiles/{name}.txt"
    birth_date = parse_birth_date(date_str)
    birth_time = parse_birth_time(time_str)

    if not birth_date or not birth_time:
        print("❌ 日期或時間格式無法解析")
        print(f"🔎 使用者輸入的 date_str: {date_str}")
        print(f"🔎 使用者輸入的 time_str: {time_str}")
        return

    lat_str, lon_str, full_address = get_city_latlon(city_name)
    if not lat_str:
        print("❌ 找不到城市位置")
        return

    dt = Datetime(birth_date, birth_time, timezone)
    pos = GeoPos(lat_str, lon_str)
    chart = Chart(dt, pos, IDs=const.LIST_OBJECTS)

    lines = []

    # ✅ 加入問答內容
    if qa_pairs:
        lines.append("📋 使用者回答：")
        for q, a in qa_pairs:
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")
        lines.append("="*40)

    # ✅ 加入星盤解讀
    lines += [
        f"📍 出生地點：{city_name} → {full_address}",
        f"🕒 出生時間：{birth_date} {birth_time} (UTC{timezone})",
        "",
        "🌌 行星位置與宮位："
    ]

    for obj in chart.objects:
        name = obj.id
        sign = obj.sign
        lon = obj.lon
        house = chart.houses.getObjectHouse(obj)
        house_num = house.id[-1] if house else "？"
        lines.append(f"🔸 {name:6s}: {sign} {lon:.2f}°，落在第 {house_num} 宮")


    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 星盤與問答資訊已儲存至 {output_path}")


# ===== 範例測試 =====
if __name__ == "__main__":
    generate_chart_from_user_input(
        date_str="2004, July 6th",
        time_str="3 PM",
        city_name="taipei city"
    )
