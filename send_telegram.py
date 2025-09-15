# send_telegram.py
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logger = logging.getLogger(__name__)

def send_telegram_message(message: str) -> None:
    """Sends a message to your Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not found. Skipping message send.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Remove Markdown parse_mode to avoid formatting errors (e.g., underscores in symbols)
    # Also truncate overly long messages just in case
    safe_text = str(message)[:3500]
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": safe_text
    }


    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram send failed ({response.status_code}): {response.text}")
            return False
        return True
    except Exception as e:
        logger.exception(f"Telegram error sending message: {e}")
        return False


def send_startup_test():
    """Send a one-time startup test message; call from main if desired."""
    return send_telegram_message("🚀 Bot startup check: Telegram channel reachable.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧪 Running Telegram test...")
    ok = send_telegram_message("✅ Telegram test message from bot.")
    print("Result:", ok)
