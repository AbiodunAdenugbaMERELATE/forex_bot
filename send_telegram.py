# send_telegram.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str) -> None:
    """Sends a message to your Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not found.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }


    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ Telegram send failed: {response.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")


if __name__ == "__main__":
    print("🧪 Running Telegram test...")
    print("TOKEN:", TELEGRAM_TOKEN)
    print("CHAT ID:", TELEGRAM_CHAT_ID)
    send_telegram_message("✅ Telegram test message from bot.")
