import requests
import base64
from app.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def send_telegram_alert(message, image_base64=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

    if image_base64:
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(
            photo_url,
            data={"chat_id": TELEGRAM_CHAT_ID},
            files={"photo": ("breakout.png", base64.b64decode(image_base64))},
        )
