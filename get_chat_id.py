"""
get_chat_id.py
==============
Run this script ONCE on your laptop to get your Telegram chat ID.
Then add that chat ID as a GitHub Secret called TELEGRAM_CHAT_ID.

Steps:
1. Open Telegram and send ANY message to your bot (e.g. "hello")
2. Run: python get_chat_id.py
3. Copy the chat_id shown
4. Go to GitHub repo → Settings → Secrets → New secret
   Name:  TELEGRAM_CHAT_ID
   Value: <the number shown>
"""

import requests

# Paste your bot token here temporarily to run this script
TOKEN = "8784948027:AAEAqpKe0j_zxy4SM7zew1oZBtum7hLDQgA"

url  = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
resp = requests.get(url, timeout=10)
data = resp.json()

if not data.get("ok"):
    print("ERROR:", data)
    exit(1)

results = data.get("result", [])
if not results:
    print("=" * 60)
    print("NO MESSAGES FOUND!")
    print("=" * 60)
    print()
    print("You need to send a message to your bot FIRST.")
    print("Steps:")
    print("  1. Open Telegram")
    print("  2. Search for your bot by name")
    print("  3. Send any message (e.g. /start or just 'hello')")
    print("  4. Then run this script again")
else:
    print("=" * 60)
    print("YOUR CHAT ID:")
    print("=" * 60)
    for r in results:
        msg     = r.get("message", {})
        chat    = msg.get("chat", {})
        chat_id = chat.get("id", "?")
        name    = chat.get("first_name", "?")
        print(f"  Name:    {name}")
        print(f"  Chat ID: {chat_id}")
        print()
        print(f"  👉  Add this as GitHub Secret 'TELEGRAM_CHAT_ID': {chat_id}")
    print("=" * 60)
