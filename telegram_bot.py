"""
Conversational Telegram Bot module for TechNews.

Handles incoming messages, questions, and commands from users:
- /start, /help: Welcome message and command guide
- /news, /digest: Trigger on-demand news fetch & categorized digest
- /search <query>: Real-time search across recent news and web
- /status: Show scheduler status, sent stories count, and configuration
- Interactive conversational chat: Discusses tech, AI, cybersecurity, and news
  using Anthropic Claude (if configured) or an intelligent built-in fallback.
"""

import asyncio
import html
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

import sqlite3

from graph import run_agent
from server import (
    DEFAULT_USER_AGENT,
    MAX_TELEGRAM_CHUNK_SIZE,
    split_telegram_message,
    web_search,
)
from state_store import DEFAULT_DB_PATH, get_sent_records

load_dotenv()
logger = logging.getLogger("telegram_bot")

TELEGRAM_GET_UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"
TELEGRAM_SEND_MESSAGE_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_SEND_ACTION_URL = "https://api.telegram.org/bot{token}/sendChatAction"


def get_db_stats() -> Dict[str, Any]:
    """Retrieve statistics from the SQLite sent.db store."""
    try:
        if not os.path.exists(DEFAULT_DB_PATH):
            return {"total_sent": 0, "recent": []}
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sent_articles")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT title, url, sent_at FROM sent_articles ORDER BY rowid DESC LIMIT 5")
            recent = [{"title": r[0] or "Article", "url": r[1], "source": "Sent", "sent_at": r[2]} for r in cursor.fetchall()]
            return {"total_sent": total, "recent": recent}
    except Exception as e:
        logger.warning("Could not fetch DB stats: %s", e)
        return {"total_sent": 0, "recent": []}


def search_recent_db_stories(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Search recently sent stories matching query words."""
    try:
        words = [w.strip() for w in re.split(r"\W+", query.lower()) if len(w.strip()) > 2]
        if not words or not os.path.exists(DEFAULT_DB_PATH):
            return []
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title, url, sent_at FROM sent_articles ORDER BY rowid DESC LIMIT 200")
            rows = cursor.fetchall()

        matches = []
        for title, url, sent_at in rows:
            if not title:
                continue
            title_lower = title.lower()
            if any(w in title_lower for w in words):
                matches.append({"title": title, "url": url, "source": "News Archive", "sent_at": sent_at})
                if len(matches) >= limit:
                    break
        return matches
    except Exception as e:
        logger.warning("Error searching recent DB stories: %s", e)
        return []


async def send_typing_action(token: str, chat_id: int | str):
    """Notify Telegram that the bot is typing a response."""
    try:
        url = TELEGRAM_SEND_ACTION_URL.format(token=token)
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json={"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


async def reply_telegram_message(
    token: str,
    chat_id: int | str,
    text: str,
    reply_to_message_id: Optional[int] = None,
) -> bool:
    """Send an HTML-formatted message to a specific chat_id with chunking."""
    if not text or not text.strip():
        return False

    url = TELEGRAM_SEND_MESSAGE_URL.format(token=token)
    chunks = split_telegram_message(text, MAX_TELEGRAM_CHUNK_SIZE)

    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, chunk in enumerate(chunks, start=1):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_to_message_id and idx == 1:
                payload["reply_to_message_id"] = reply_to_message_id

            try:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    safe_body = resp.text.replace(token, "[REDACTED]")
                    logger.error("Telegram send error (status %d): %s", resp.status_code, safe_body)
                    # If HTML parse error occurred, retry once as plain text
                    if "can't parse entities" in resp.text.lower():
                        payload["parse_mode"] = ""
                        del payload["parse_mode"]
                        payload["text"] = html.unescape(chunk)
                        await client.post(url, json=payload)
            except Exception as e:
                safe_err = str(e).replace(token, "[REDACTED]")
                logger.error("Failed to send reply to chat %s: %s", chat_id, safe_err)
                return False

    return True


async def generate_conversational_reply(user_message: str, user_name: str = "Friend") -> str:
    """Generate an intelligent tech response using Claude or contextual fallback."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()

    # If Anthropic Claude API key is configured, use it for unconstrained reasoning
    if anthropic_key and not anthropic_key.startswith("your_"):
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatAnthropic(
                model=model_name,
                api_key=anthropic_key,
                temperature=0.7,
                max_tokens=600,
            )
            system_prompt = (
                "You are TechNews Assistant, an intelligent, conversational tech and AI companion inside a Telegram bot. "
                "You are knowledgeable, engaging, balanced, and articulate about computer science, artificial intelligence, "
                "cybersecurity, hardware, robotics, and the tech industry. "
                "Keep your response concise, sharp, and conversational (typically 2-3 short paragraphs or bullet points). "
                "Format using Telegram HTML (<b>bold</b>, <i>italic</i>, <code>code</code>, <a href='...'>links</a>). "
                "Always escape all literal ampersands as &amp;. Do not wrap the response in markdown backticks."
            )
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User ({user_name}) said: {user_message}"),
            ])
            text = response.content if isinstance(response.content, str) else str(response.content)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning("Claude conversation call failed, falling back to smart responder: %s", e)

    # Contextual Intelligent Fallback Engine
    msg_lower = user_message.lower().strip()

    # 1. AI pace / progress / concern (e.g. "ai is getting to far", "ai is crazy", "ai taking over")
    if any(k in msg_lower for k in ["too far", "to far", "getting far", "scary", "taking over", "crazy", "fast", "future of ai", "agi"]):
        return (
            "🤖 <b>AI is indeed progressing at a breathtaking pace!</b>\n\n"
            "From reasoning models that deliberate before answering, to autonomous agents using developer tools "
            "and executing multi-step workflows, the rate of capability jumps in recent months has been unprecedented.\n\n"
            "<b>Key frontiers moving fastest right now:</b>\n"
            "• <b>Reasoning &amp; Test-Time Compute</b>: Shifting from fast next-token prediction to deep architectural deliberation.\n"
            "• <b>Autonomous Tool Use &amp; MCP</b>: Agents interacting directly with codebases, APIs, and browsers.\n"
            "• <b>Physical &amp; Embodied AI</b>: Rapid progress in robotics and industrial automation.\n\n"
            "<i>What specific breakthrough or trend has caught your attention most? You can also type /news to see today's top tech releases!</i>"
        )

    # 2. Cybersecurity & hacking inquiries
    if any(k in msg_lower for k in ["hack", "breach", "cyber", "security", "malware", "ransomware", "vulnerability", "leak"]):
        recent_security = search_recent_db_stories("cyber security hack breach vulnerability", limit=3)
        rec_text = ""
        if recent_security:
            rec_text = "\n\n<b>Recent cybersecurity headlines recorded:</b>\n" + "\n".join(
                f"• <a href='{s['url']}'>{html.escape(s['title'])}</a> ({s['source']})" for s in recent_security
            )
        return (
            "🛡️ <b>Cybersecurity is in a state of rapid transformation.</b>\n\n"
            "With the rise of AI-assisted exploits and automated vulnerability scanning, defense teams are "
            "shifting toward zero-trust architectures, automated patching, and memory-safe programming languages (like Rust)."
            f"{rec_text}\n\n"
            "<i>Ask me about any specific security CVE, company breach, or use <code>/search &lt;query&gt;</code> to dig deeper!</i>"
        )

    # 3. Hardware / Semiconductor / Chips
    if any(k in msg_lower for k in ["chip", "gpu", "nvidia", "hardware", "intel", "amd", "tsmc", "quantum"]):
        recent_hw = search_recent_db_stories("chip gpu nvidia intel hardware quantum tsmc", limit=3)
        rec_text = ""
        if recent_hw:
            rec_text = "\n\n<b>Recent hardware stories in our digest:</b>\n" + "\n".join(
                f"• <a href='{s['url']}'>{html.escape(s['title'])}</a> ({s['source']})" for s in recent_hw
            )
        return (
            "⚡ <b>Hardware &amp; Silicon Innovation:</b>\n\n"
            "The race for compute efficiency is fiercer than ever. Between advanced 2nm/3nm packaging, high-bandwidth "
            "memory (HBM3e/HBM4), and specialized neural processing units, hardware remains the foundation of all modern software scale."
            f"{rec_text}\n\n"
            "<i>Type <code>/search &lt;topic&gt;</code> to look up latest hardware benchmarks or company earnings!</i>"
        )

    # 4. Direct questions about recent news or web search
    if any(msg_lower.startswith(q) for q in ["what is ", "who is ", "how does ", "why did ", "tell me about ", "explain "]):
        search_topic = re.sub(r"^(what is|who is|how does|why did|tell me about|explain)\s+", "", msg_lower).strip(" ?.")
        if search_topic:
            try:
                raw_search = await web_search(search_topic, limit=3)
                import json
                search_data = json.loads(raw_search)
                results = search_data.get("results", [])
                if results:
                    summary_lines = [
                        f"• <b>{html.escape(r.get('title', ''))}</b>: {html.escape(r.get('snippet', '')[:160])}..."
                        for r in results if r.get("snippet")
                    ]
                    return (
                        f"🔍 <b>Insights on '{html.escape(search_topic)}':</b>\n\n"
                        + "\n\n".join(summary_lines)
                        + "\n\n<i>Use <code>/search &lt;topic&gt;</code> for more results or ask another question!</i>"
                    )
            except Exception:
                pass

    # 5. Default friendly conversational response
    recent_db = search_recent_db_stories(msg_lower, limit=2)
    extra_context = ""
    if recent_db:
        extra_context = "\n\n<b>Related recent stories:</b>\n" + "\n".join(
            f"• <a href='{s['url']}'>{html.escape(s['title'])}</a>" for s in recent_db
        )

    return (
        f"👋 <b>Thanks for your message, {html.escape(user_name)}!</b>\n\n"
        f"I'm keeping a constant pulse on breaking tech news, AI research, and cybersecurity developments."
        f"{extra_context}\n\n"
        "Here are a few quick things you can do:\n"
        "• Type <b>/news</b> to get the latest categorized tech digest immediately.\n"
        "• Type <b>/search &lt;topic&gt;</b> (e.g. <code>/search deepseek</code>) to research any subject.\n"
        "• Type <b>/status</b> to check pipeline and database stats.\n"
        "• Or ask me any question about technology and AI!"
    )


async def handle_user_command(command: str, args: str, chat_id: int | str, user_name: str) -> str:
    """Handle slash commands like /start, /help, /news, /status, /search."""
    cmd = command.lower().strip()

    if cmd in ("/start", "/help"):
        return (
            f"👋 <b>Hello {html.escape(user_name)}! Welcome to TechNews Bot.</b>\n\n"
            "I am your automated AI-driven tech news and research assistant.\n\n"
            "<b>Available Commands:</b>\n"
            "📰 <b>/news</b> or <b>/digest</b> — Fetch latest breaking stories and send digest now\n"
            "🔍 <b>/search &lt;query&gt;</b> — Search breaking tech news &amp; web docs (e.g. <code>/search nvidia</code>)\n"
            "📊 <b>/status</b> — View pipeline status, database stats, and schedule\n"
            "🏷️ <b>/categories</b> — View our 5 news topic categories\n"
            "ℹ️ <b>/help</b> — Show this guide\n\n"
            "💬 <b>Chat with me:</b>\n"
            "You can talk to me directly about any technology news, AI advancement, or questions. I'm listening 24/7!"
        )

    if cmd in ("/news", "/digest"):
        return "__TRIGGER_NEWS_PIPELINE__"

    if cmd in ("/categories", "/topics"):
        return (
            "🏷️ <b>TechNews Categories:</b>\n\n"
            "🤖 <b>New Models &amp; AI</b> — LLMs, foundation models, weights, fine-tuning, benchmarks\n"
            "💡 <b>New Innovation</b> — Breakthrough discoveries, algorithms, quantum, clean energy tech\n"
            "🛡️ <b>Cyber Crime &amp; Security</b> — Breaches, zero-days, CVEs, ransomware, privacy\n"
            "⚡ <b>New Tech &amp; Hardware</b> — Chips, semiconductors, robotics, gadgets, space hardware\n"
            "🌐 <b>Industry &amp; Big Tech</b> — Major policy, acquisitions, investments, market shifts\n\n"
            "<i>We aggregate across Hacker News, TechCrunch, The Verge, Ars Technica, The Guardian, MIT Tech Review, and DeepLearning.AI!</i>"
        )

    if cmd == "/status":
        stats = get_db_stats()
        total_stories = stats.get("total_sent", 0)
        recent = stats.get("recent", [])
        recent_str = ""
        if recent:
            recent_str = "\n<b>Recently dispatched:</b>\n" + "\n".join(
                f"• {html.escape(r['title'][:55])}... (<i>{r['source']}</i>)" for r in recent[:3]
            )

        interval = os.getenv("CHECK_INTERVAL_MINUTES", "60")
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        return (
            "📊 <b>TechNews System Status:</b>\n\n"
            f"• <b>Status</b>: 🟢 Active &amp; Operational\n"
            f"• <b>Recurring Schedule</b>: Every {interval} minutes\n"
            f"• <b>Total Unique Stories Sent</b>: <code>{total_stories}</code>\n"
            f"• <b>Summarizer Mode</b>: {'Claude (' + model + ')' if has_anthropic else 'Smart Structured Extractor'}\n"
            f"• <b>Active Sources (7)</b>: Hacker News, TechCrunch, The Verge, Ars Technica, Guardian, MIT Review, DeepLearning.AI\n"
            f"{recent_str}\n\n"
            "<i>Everything is running smoothly!</i>"
        )

    if cmd == "/search":
        query = args.strip()
        if not query:
            return "🔍 <i>Please specify what you would like to search for.</i>\n\nExample:\n<code>/search deepseek v3</code>\n<code>/search quantum computing breakthrough</code>"

        import json
        raw_res = await web_search(query, limit=4)
        try:
            data = json.loads(raw_res)
            results = data.get("results", [])
            if not results:
                return f"🔍 No recent news or web results found for '<b>{html.escape(query)}</b>'."

            formatted = [f"🔍 <b>Search results for '{html.escape(query)}':</b>\n"]
            for r in results:
                title = html.escape(r.get("title", "Untitled"))
                snippet = html.escape(r.get("snippet", ""))
                url = r.get("url", "")
                formatted.append(f"• <a href='{url}'><b>{title}</b></a>\n  {snippet}\n")

            return "\n".join(formatted)
        except Exception as e:
            logger.error("Search error: %s", e)
            return f"❌ An error occurred during search: {html.escape(str(e))}"

    return ""


async def process_telegram_update(update: Dict[str, Any], token: str):
    """Process an individual update dictionary from Telegram getUpdates."""
    message = update.get("message")
    if not message:
        return

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    if not chat_id:
        return

    text = message.get("text", "").strip()
    user_info = message.get("from", {})
    user_name = user_info.get("first_name") or user_info.get("username") or "Friend"
    message_id = message.get("message_id")

    if not text:
        return

    logger.info("📩 Message from @%s (%s): %s", user_info.get("username", "user"), chat_id, text)

    # Show typing indicator
    await send_typing_action(token, chat_id)

    # Check for slash commands
    reply_text = ""
    if text.startswith("/"):
        parts = text.split(" ", 1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        # Special handler for on-demand news trigger
        if command.lower() in ("/news", "/digest"):
            await reply_telegram_message(
                token,
                chat_id,
                "⏳ <b>Fetching, filtering, and researching the latest tech news...</b>\n<i>Please hold on a few seconds!</i>",
                reply_to_message_id=message_id,
            )
            try:
                res = await run_agent()
                status = res.get("status", "")
                if "No new stories" in status:
                    reply_text = (
                        "ℹ️ <b>No brand new stories published since the last digest.</b>\n\n"
                        "All 7 publications (Hacker News, TechCrunch, The Verge, Ars Technica, Guardian, MIT Review, DeepLearning.AI) "
                        "have already been delivered to your chat. We will deliver new stories as soon as they break!"
                    )
                else:
                    reply_text = f"✅ <b>Done!</b> {html.escape(status)}"
            except Exception as e:
                reply_text = f"❌ Error running news digest: {html.escape(str(e))}"
        else:
            reply_text = await handle_user_command(command, args, chat_id, user_name)

    # Normal conversational chat
    if not reply_text:
        reply_text = await generate_conversational_reply(text, user_name)

    if reply_text:
        await reply_telegram_message(token, chat_id, reply_text, reply_to_message_id=message_id)


async def poll_telegram_updates(stop_event: Optional[asyncio.Event] = None):
    """Continuously poll Telegram getUpdates endpoint for incoming chats."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token.startswith("your_"):
        logger.warning("TELEGRAM_BOT_TOKEN not configured; Telegram chat listener disabled.")
        return

    logger.info("💬 Telegram Chatbot polling listener started. Ready to chat!")
    offset: Optional[int] = None

    async with httpx.AsyncClient(timeout=35.0) as client:
        while stop_event is None or not stop_event.is_set():
            try:
                params = {"timeout": 20}
                if offset is not None:
                    params["offset"] = offset

                url = TELEGRAM_GET_UPDATES_URL.format(token=token)
                resp = await client.get(url, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    updates = data.get("result", [])
                    for update in updates:
                        up_id = update.get("update_id")
                        if up_id is not None:
                            offset = up_id + 1
                        try:
                            await process_telegram_update(update, token)
                        except Exception as e:
                            logger.error("Error processing update %s: %s", up_id, e)

                elif resp.status_code == 409:
                    # Another instance is polling (e.g. dev reload or second process)
                    logger.warning("Telegram getUpdates 409 Conflict: another instance is polling. Waiting 15s...")
                    await asyncio.sleep(15)
                else:
                    safe_text = resp.text.replace(token, "[REDACTED]")
                    logger.warning("Telegram getUpdates HTTP %d: %s", resp.status_code, safe_text)
                    await asyncio.sleep(5)

            except asyncio.CancelledError:
                logger.info("Telegram polling task cancelled.")
                break
            except Exception as e:
                safe_err = str(e).replace(token, "[REDACTED]")
                logger.warning("Transient error in Telegram polling: %s. Retrying in 5s...", safe_err)
                await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(poll_telegram_updates())
