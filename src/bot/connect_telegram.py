"""
One-click utility to connect your Telegram bot, capture your Chat ID, and send a test message.
"""

import os
import sys
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "8548609073:AAG-mIOVCh-8ZHCPphkUvYRY3GztkNcdp8Q"
BOT_USERNAME = "daily_techdigest_bot"


def main():
    print("=" * 60)
    print("  Telegram Bot Connection Wizard")
    print(f"  Bot: @{BOT_USERNAME}")
    print("=" * 60)

    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

    # Step 1: Check existing updates
    try:
        resp = httpx.get(url, timeout=10.0)
        data = resp.json()
    except Exception as e:
        print(f"❌ Error communicating with Telegram API: {e}")
        return

    chat_id = None
    if data.get("ok") and data.get("result"):
        # Look backwards for the most recent message
        for update in reversed(data["result"]):
            msg = update.get("message") or update.get("channel_post") or update.get("my_chat_member")
            if msg and "chat" in msg:
                chat_id = str(msg["chat"]["id"])
                username = msg["chat"].get("username", "Unknown")
                print(f"✅ Detected chat from @{username} (ID: {chat_id})")
                break

    if not chat_id:
        print("\n👉 Next Step:")
        print(f"1. Open Telegram and search for: @{BOT_USERNAME}")
        print(f"   Or click link: https://t.me/{BOT_USERNAME}")
        print("2. Click START (or send /start or 'hi' to the bot).")
        print("\nWaiting up to 45 seconds for your message...")

        for remaining in range(45, 0, -3):
            time.sleep(3)
            try:
                resp = httpx.get(url, timeout=10.0)
                d = resp.json()
                if d.get("ok") and d.get("result"):
                    for update in reversed(d["result"]):
                        msg = update.get("message") or update.get("channel_post")
                        if msg and "chat" in msg:
                            chat_id = str(msg["chat"]["id"])
                            print(f"\n🎉 Message received! Chat ID: {chat_id}")
                            break
                if chat_id:
                    break
                print(f"  ...waiting ({remaining}s remaining)")
            except Exception:
                pass

    if not chat_id:
        print("\n⏳ No message received yet.")
        print(f"Please message @{BOT_USERNAME} and run this script again:")
        print("    python connect_telegram.py")
        return

    # Step 2: Update .env with TELEGRAM_CHAT_ID
    env_lines = []
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            env_lines = f.readlines()

    updated = False
    new_lines = []
    for line in env_lines:
        if line.startswith("TELEGRAM_CHAT_ID="):
            new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}\n")
            updated = True
        elif line.startswith("TELEGRAM_BOT_TOKEN="):
            new_lines.append(f"TELEGRAM_BOT_TOKEN={TOKEN}\n")
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}\n")

    with open(".env", "w") as f:
        f.writelines(new_lines)
    print("✅ Updated .env with your TELEGRAM_CHAT_ID.")

    # Step 3: Send test confirmation message
    send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": (
            "🎉 <b>Bot Connected Successfully!</b>\n\n"
            "Your <b>tech-news-mcp</b> pipeline is now connected to this chat.\n"
            "You can trigger digests anytime by running <code>python graph.py</code>."
        ),
        "parse_mode": "HTML",
    }
    try:
        r = httpx.post(send_url, json=payload, timeout=10.0)
        if r.status_code == 200:
            print("🚀 Confirmation message sent to your Telegram!")
        else:
            print(f"⚠️ Telegram response ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"⚠️ Failed to send test message: {e}")


if __name__ == "__main__":
    main()

