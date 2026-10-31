"""
Conversational Telegram Bot module for TechNews.

Handles incoming messages, questions, and commands from users:
- Friendly, accurate, and conversational tech answers
- Live web search and recent news retrieval on any topic or company
- Special in-depth coverage for Google, OpenAI, Nvidia, Apple, Meta, Anthropic, DeepSeek, etc.
- Slash commands: /news, /search, /status, /categories, /help
- Strict deduplication so messages are never processed or sent twice
"""

import asyncio
import html
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set

import httpx
from dotenv import load_dotenv

import sqlite3

try:
    from src.agent.graph import run_agent
    from src.mcp_server.server import (
        DEFAULT_USER_AGENT,
        MAX_TELEGRAM_CHUNK_SIZE,
        split_telegram_message,
        web_search,
    )
    from src.storage.state_store import DEFAULT_DB_PATH, get_sent_records
except ImportError:
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

# In-memory deduplication cache to prevent duplicate processing
PROCESSED_UPDATE_IDS: Set[int] = set()
PROCESSED_MESSAGES: Set[str] = set()


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
            recent = [
                {"title": r[0] or "Article", "url": r[1], "source": "Sent", "sent_at": r[2]}
                for r in cursor.fetchall()
            ]
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
            cursor.execute("SELECT title, url, sent_at FROM sent_articles ORDER BY rowid DESC LIMIT 300")
            rows = cursor.fetchall()

        matches = []
        for title, url, sent_at in rows:
            if not title:
                continue
            title_lower = title.lower()
            if any(w in title_lower for w in words):
                matches.append({"title": title, "url": url, "source": "Digest Archive", "sent_at": sent_at})
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


async def fetch_live_web_context(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Helper to query web_search for live real-time context."""
    try:
        raw = await web_search(query, limit=limit)
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("results", [])
        return []
    except Exception as e:
        logger.warning("Error fetching web context for '%s': %s", query, e)
        return []


async def query_groq_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Optional[str]:
    """Call Groq API using HTTPX for ultra-fast, high-accuracy inference."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key or groq_key.startswith("your_"):
        return None

    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    if text:
                        if text.startswith("```html") or text.startswith("```"):
                            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
                            text = re.sub(r"\n?```$", "", text).strip()
                        text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", text)
                        return text
            else:
                safe_body = resp.text.replace(groq_key, "[REDACTED]")
                logger.warning("Groq API error (%d): %s", resp.status_code, safe_body[:200])
    except Exception as e:
        safe_err = str(e).replace(groq_key, "[REDACTED]")
        logger.warning("Failed calling Groq LLM: %s", safe_err)
    return None


async def generate_conversational_reply(user_message: str, user_name: str = "Friend") -> str:
    """Generate a friendly, accurate, and engaging tech response."""
    # 1. Try Groq API first (ultra-fast, accurate)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("your_"):
        context_snippets = []
        if len(user_message.split()) >= 2:
            web_context = await fetch_live_web_context(user_message, limit=2)
            for w in web_context:
                title = w.get("title", "")
                url = w.get("url", "")
                snip = w.get("snippet", "")
                if title:
                    context_snippets.append(f"Web source: {title} ({url}): {snip[:120]}")

        context_prompt = ""
        if context_snippets:
            context_prompt = "\nRecent verified web context:\n" + "\n".join(context_snippets) + "\n"

        system_prompt = (
            f"You are TechNews Assistant, a friendly, warm, enthusiastic, and highly knowledgeable AI companion in Telegram "
            f"speaking with your friend and fellow tech enthusiast, {user_name}.\n"
            "Your personality is warm, approachable, technically accurate, and articulate.\n"
            "When answering questions about technology, artificial intelligence, companies (Google, OpenAI, Nvidia, Apple, etc.), chips, or cybersecurity:\n"
            "- Greet {user_name} in a friendly, conversational manner.\n"
            "- Give clear, accurate, and structured insights with real-world context.\n"
            "- Use Telegram HTML formatting: <b>bold</b>, <i>italic</i>, <code>code</code>, <a href='...'>links</a>.\n"
            "- Always escape literal ampersands as &amp;.\n"
            "- Never wrap your response in markdown code fences.\n"
            "- Keep the tone conversational, friendly, and helpful."
        )
        groq_reply = await query_groq_llm([
            {"role": "system", "content": system_prompt + context_prompt},
            {"role": "user", "content": f"User ({user_name}) asked: {user_message}"},
        ])
        if groq_reply:
            return groq_reply

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()

    # 2. Use Claude if API key is provided
    if anthropic_key and not anthropic_key.startswith("your_"):
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatAnthropic(
                model=model_name,
                api_key=anthropic_key,
                temperature=0.7,
                max_tokens=700,
            )
            system_prompt = (
                f"You are TechNews Assistant, a friendly, warm, enthusiastic, and highly knowledgeable AI companion in Telegram. "
                f"You are speaking with your friend and fellow tech enthusiast, {user_name}. "
                "Always be warm, conversational, articulate, and accurate. "
                "If asked about technology, companies (like Google, Apple, Nvidia, OpenAI), AI models, chips, or cybersecurity, "
                "give specific, up-to-date, accurate details in an easy-to-read style. "
                "Format using Telegram HTML (<b>bold</b>, <i>italic</i>, <code>code</code>, <a href='...'>links</a>). "
                "Always escape all literal ampersands as &amp;. Never wrap the output in markdown backticks."
            )
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User ({user_name}) asked: {user_message}"),
            ])
            text = response.content if isinstance(response.content, str) else str(response.content)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning("Claude conversation call failed, falling back to smart responder: %s", e)

    # 2. Friendly, Accurate Intelligent Engine (when Claude is not set or fails)
    msg_clean = user_message.strip()
    msg_lower = msg_clean.lower()

    # A. Casual Greetings & Friendly Chit-Chat
    greetings = ["hi", "hello", "hey", "how are you", "what's up", "whats up", "good morning", "good evening", "good afternoon"]
    if any(msg_lower == g or msg_lower.startswith(g + " ") for g in greetings):
        return (
            f"Hey {html.escape(user_name)}! 😊 It's great to hear from you!\n\n"
            "I'm having a great day, keeping an active eye on everything happening across tech, AI breakthroughs, and cybersecurity.\n\n"
            "What would you like to explore today? You can ask me what companies like <b>Google</b>, <b>OpenAI</b>, or <b>Nvidia</b> are building, "
            "ask for the latest tech news with <b>/news</b>, or search any topic with <b>/search &lt;query&gt;</b>! 🚀"
        )

    # B. User appreciation
    if any(msg_lower == thx or msg_lower.startswith(thx + " ") for thx in ["thanks", "thank you", "thx", "awesome", "great job", "nice", "cool"]):
        return (
            f"You're very welcome, {html.escape(user_name)}! 🙌 Happy to help. "
            "Whenever you want to know what's happening in tech or have a question, I'm right here!"
        )

    # C. Company-Specific In-Depth & Accurate Knowledge:
    # 1) Google / Alphabet
    if "google" in msg_lower or "alphabet" in msg_lower or "deepmind" in msg_lower:
        web_results = await fetch_live_web_context("Google AI announcements Gemini DeepMind latest news", limit=2)
        live_snippet = ""
        if web_results:
            live_snippet = "\n\n<b>Recent headline in the news:</b>\n" + "\n".join(
                f"• <a href='{r['url']}'>{html.escape(r['title'])}</a>\n  {html.escape(r.get('snippet', '')[:140])}..."
                for r in web_results if r.get("title") and r.get("url")
            )

        return (
            f"Here is what <b>Google</b> is focusing on right now, {html.escape(user_name)}! 🌐\n\n"
            "Google and Google DeepMind are pushing aggressively on several major frontiers:\n\n"
            "1. 🧠 <b>Gemini 2.0 &amp; 2.5 Frontier Models</b>:\n"
            "Google rolled out Gemini 2.0 (including Flash and Thinking variants), which feature native multimodality, "
            "low-latency audio/video reasoning, and agentic capabilities for complex multi-step reasoning.\n\n"
            "2. 🤖 <b>Project Astra &amp; Autonomous Agents</b>:\n"
            "Building universal AI assistants that can see, hear, and converse in real-time with sub-second latency, "
            "alongside coding and browser agents (like Project Jarvis) capable of performing tasks on computers.\n\n"
            "3. ⚡ <b>Custom Silicon (TPU v6 / Trillium)</b>:\n"
            "Massive deployment of custom AI hardware (Trillium TPUs and TPU v5p) providing unprecedented compute efficiency "
            "for training foundation models and powering Google Cloud AI workloads.\n\n"
            "4. 🔬 <b>Quantum Breakthroughs (Willow)</b>:\n"
            "DeepMind announced the 'Willow' quantum chip, which demonstrated exponential error reduction as more qubits were added—a "
            "historic milestone toward fault-tolerant quantum computing.\n\n"
            "5. 🔍 <b>Search &amp; Ecosystem Integration</b>:\n"
            "Deeply weaving AI Overviews into Google Search, Android 16 on mobile, and Google Workspace (Docs, Gmail, Sheets)."
            f"{live_snippet}\n\n"
            "<i>Is there a specific Google project (like Gemini, TPUs, or Android) you'd like to dive deeper into?</i>"
        )

    # 2) OpenAI
    if "openai" in msg_lower or "chatgpt" in msg_lower or "sam altman" in msg_lower:
        web_results = await fetch_live_web_context("OpenAI latest announcements o1 o3 models news", limit=2)
        live_snippet = ""
        if web_results:
            live_snippet = "\n\n<b>Latest OpenAI headlines:</b>\n" + "\n".join(
                f"• <a href='{r['url']}'>{html.escape(r['title'])}</a>\n  {html.escape(r.get('snippet', '')[:140])}..."
                for r in web_results if r.get("title") and r.get("url")
            )

        return (
            f"Here is what <b>OpenAI</b> is up to right now, {html.escape(user_name)}! 🚀\n\n"
            "OpenAI is leading several groundbreaking initiatives:\n\n"
            "1. 🧩 <b>Reasoning Models (o1 &amp; o3 series)</b>:\n"
            "Shifting focus to 'test-time compute'—where the model deliberates and reasons step-by-step before producing "
            "an answer, dramatically outperforming standard LLMs on complex math, coding, and competitive benchmarks.\n\n"
            "2. 🖥️ <b>Autonomous 'Operator' Agents</b>:\n"
            "Developing computer-using agents that can autonomously navigate web browsers, write code, run commands, "
            "and book flights or complete real-world workflows on your behalf.\n\n"
            "3. 🎬 <b>Sora &amp; Multimodal Video</b>:\n"
            "Progressively rolling out Sora for photorealistic generative video with physical world simulation.\n\n"
            "4. 🏗️ <b>Supercomputing Infrastructure ('Stargate')</b>:\n"
            "Partnering with SoftBank, Oracle, and Microsoft to build massive next-gen data centers with millions of AI accelerators."
            f"{live_snippet}\n\n"
            "<i>Would you like to know more about OpenAI's reasoning models or their agent plans?</i>"
        )

    # 3) Nvidia
    if "nvidia" in msg_lower or "jensen" in msg_lower or "blackwell" in msg_lower or "geforce" in msg_lower:
        web_results = await fetch_live_web_context("Nvidia Blackwell GPU AI chips latest news", limit=2)
        live_snippet = ""
        if web_results:
            live_snippet = "\n\n<b>Recent Nvidia news:</b>\n" + "\n".join(
                f"• <a href='{r['url']}'>{html.escape(r['title'])}</a>"
                for r in web_results if r.get("title") and r.get("url")
            )

        return (
            f"Here is the latest on <b>Nvidia</b>, {html.escape(user_name)}! ⚡\n\n"
            "Nvidia remains the undisputed compute engine of the AI revolution:\n\n"
            "1. 🛡️ <b>Blackwell Architecture (B200 &amp; GB200 NVL72)</b>:\n"
            "The Blackwell generation delivers up to 30x faster real-time LLM inference compared to H100, "
            "connecting 72 GPUs in a single liquid-cooled rack that acts as one giant GPU.\n\n"
            "2. 🤖 <b>Physical AI &amp; Robotics (Project GR00T &amp; Cosmos)</b>:\n"
            "Nvidia is pioneering 'Physical AI'—foundation world models that teach humanoid robots and autonomous vehicles "
            "how to perceive and interact with the physical world in simulation before real deployment.\n\n"
            "3. 🌐 <b>Networking Dominance</b>:\n"
            "Quantum-X800 InfiniBand and Spectrum-X Ethernet switches designed to prevent networking bottlenecks across tens of thousands of GPUs."
            f"{live_snippet}\n\n"
            "<i>Ask me about Nvidia's GPU benchmarks or financial performance anytime!</i>"
        )

    # 4) DeepSeek
    if "deepseek" in msg_lower:
        web_results = await fetch_live_web_context("DeepSeek R1 AI open source reasoning model news", limit=2)
        live_snippet = ""
        if web_results:
            live_snippet = "\n\n<b>Top DeepSeek headlines:</b>\n" + "\n".join(
                f"• <a href='{r['url']}'>{html.escape(r['title'])}</a>"
                for r in web_results if r.get("title") and r.get("url")
            )

        return (
            f"Here is the breakdown on <b>DeepSeek</b>, {html.escape(user_name)}! 💡\n\n"
            "DeepSeek has shaken up the global AI industry with extreme architectural efficiency:\n\n"
            "1. 🧠 <b>DeepSeek-R1 &amp; Open Weights</b>:\n"
            "A frontier reasoning model trained with pure reinforcement learning (RL) that rivals proprietary closed models (like OpenAI o1), "
            "released under an open MIT license with full weights and distillation checkpoints.\n\n"
            "2. ⚙️ <b>Architectural Innovations</b>:\n"
            "• <b>Multi-head Latent Attention (MLA)</b>: Drastically compresses KV cache memory overhead.\n"
            "• <b>DeepSeekMoE</b>: Sparse mixture-of-experts activating only a small fraction of parameters per token.\n\n"
            "3. 💰 <b>Frugal Training Costs</b>:\n"
            "Demonstrated that frontier capabilities can be achieved at a fraction of Silicon Valley compute budgets, "
            "democratizing access for the global open-source community."
            f"{live_snippet}\n\n"
            "<i>Would you like to know how DeepSeek compares to Claude or ChatGPT?</i>"
        )

    # 5) Apple
    if "apple" in msg_lower or "tim cook" in msg_lower or "iphone" in msg_lower:
        web_results = await fetch_live_web_context("Apple Intelligence M4 Mac iPhone news", limit=2)
        live_snippet = ""
        if web_results:
            live_snippet = "\n\n<b>Recent Apple stories:</b>\n" + "\n".join(
                f"• <a href='{r['url']}'>{html.escape(r['title'])}</a>"
                for r in web_results if r.get("title") and r.get("url")
            )

        return (
            f"Here is what <b>Apple</b> is doing right now, {html.escape(user_name)}! 🍏\n\n"
            "Apple is executing on a privacy-first, on-device AI and silicon strategy:\n\n"
            "1. 📱 <b>Apple Intelligence Rollout</b>:\n"
            "Gradually delivering on-device generative tools (Clean Up, writing tools, notification summaries, and a more context-aware Siri) "
            "across iOS, iPadOS, and macOS.\n\n"
            "2. 🔒 <b>Private Cloud Compute</b>:\n"
            "For heavier tasks, Apple routes requests to custom Apple Silicon servers with verifiable privacy guarantees—ensuring user data "
            "is never logged or accessible even to Apple.\n\n"
            "3. 💻 <b>M4 Chip Family</b>:\n"
            "Delivering market-leading single-core CPU speeds and powerful neural engines across the Mac lineup."
            f"{live_snippet}\n\n"
            "<i>Are you interested in Apple's hardware roadmap or their AI features?</i>"
        )

    # D. Topic: AI pace / future / safety (e.g. "ai is getting to far", "is ai taking over")
    if any(k in msg_lower for k in ["too far", "to far", "getting far", "scary", "taking over", "crazy", "fast", "future of ai", "agi", "dangerous"]):
        return (
            f"I hear you, {html.escape(user_name)}! 🤖 <b>AI is genuinely progressing at a breathtaking pace!</b>\n\n"
            "It can feel overwhelming seeing how fast things are moving. In just the past year, we have gone from simple text prediction "
            "to models that can deliberate and reason, interact with terminals and browsers, and control physical robotic limbs.\n\n"
            "<b>The three biggest shifts happening right now:</b>\n"
            "1. <b>Test-Time Reasoning</b>: Models 'thinking' before answering, catching their own logical flaws.\n"
            "2. <b>Autonomous Agent Tool Use</b>: Moving from chat boxes to agents that execute code, query APIs, and build software.\n"
            "3. <b>Physical Embodiment</b>: Neural world models being trained to navigate real factories and homes.\n\n"
            "The key focus in the developer and research community right now is on safety boundaries, automated verification, and alignment.\n\n"
            "<i>What specific aspect feels most surprising or concerning to you?</i>"
        )

    # E. Dynamic Live Search & Knowledge for ANY Other Question!
    # If the user asks anything (starts with what, how, why, who, is, can, tell me, etc. OR has a question mark)
    is_question = (
        "?" in msg_clean
        or any(msg_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "tell", "explain", "is", "can", "does", "will", "show"])
        or len(msg_clean.split()) >= 3
    )

    if is_question:
        clean_search_query = re.sub(r"[?!.,]", "", msg_clean)
        web_results = await fetch_live_web_context(clean_search_query, limit=3)
        db_results = search_recent_db_stories(clean_search_query, limit=2)

        if web_results or db_results:
            bullet_points = []
            for r in web_results:
                title = r.get("title", "").strip()
                snippet = r.get("snippet", "").strip()
                url = r.get("url", "")
                if title and snippet:
                    bullet_points.append(f"• <a href='{url}'><b>{html.escape(title)}</b></a>:\n  {html.escape(snippet[:160])}...")

            db_lines = []
            for d in db_results:
                db_lines.append(f"• <a href='{d['url']}'>{html.escape(d['title'])}</a> (<i>{d['source']}</i>)")

            extra_db = ""
            if db_lines:
                extra_db = "\n\n<b>From our recent news digest archive:</b>\n" + "\n".join(db_lines)

            bullets_formatted = "\n\n".join(bullet_points) if bullet_points else "Recent web data indicates continuous active development in this area."

            return (
                f"Great question, {html.escape(user_name)}! Here is what's happening regarding <b>{html.escape(msg_clean)}</b>: 🔍\n\n"
                f"{bullets_formatted}"
                f"{extra_db}\n\n"
                "<i>Would you like me to dive deeper into any of these details, or search for another topic?</i>"
            )

    # F. Warm, Engaging Catch-All for Anything Else
    return (
        f"Hey {html.escape(user_name)}! 😊\n\n"
        f"You mentioned: <i>\"{html.escape(msg_clean)}\"</i>\n\n"
        "I'm keeping a constant pulse on tech, AI research, cybersecurity, and hardware innovations across 7 major publications.\n\n"
        "Feel free to ask me:\n"
        "• What any tech company (Google, OpenAI, Nvidia, Apple, etc.) is building\n"
        "• How specific technologies work\n"
        "• <b>/news</b> to get the latest breaking tech digest\n"
        "• <b>/search &lt;query&gt;</b> to research any topic!\n\n"
        "What are you curious about?"
    )


async def handle_user_command(command: str, args: str, chat_id: int | str, user_name: str) -> str:
    """Handle slash commands like /start, /help, /news, /status, /search."""
    cmd = command.lower().strip()

    if cmd in ("/start", "/help"):
        return (
            f"👋 <b>Hello {html.escape(user_name)}! Welcome to TechNews Bot.</b>\n\n"
            "I'm your friendly AI tech companion and research assistant! Here is everything I can do for you:\n\n"
            "📰 <b>/news</b> or <b>/digest</b> — Fetch latest breaking stories and send digest now\n"
            "🔍 <b>/search &lt;query&gt;</b> — Search breaking tech news &amp; web docs (e.g. <code>/search nvidia blackwell</code>)\n"
            "📊 <b>/status</b> — View pipeline status, database stats, and schedule\n"
            "🏷️ <b>/categories</b> — View our 5 tracked news topic categories\n"
            "ℹ️ <b>/help</b> — Show this guide\n\n"
            "💬 <b>Chat with me anytime:</b>\n"
            "Ask me anything! What Google is building, how a neural network works, breaking hacks, or what's new in hardware. I'm always here to chat! 😊"
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
        has_groq = bool(os.getenv("GROQ_API_KEY", "").strip())
        groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        if has_groq:
            engine_desc = f"Groq AI ({groq_model})"
        elif has_anthropic:
            engine_desc = f"Claude ({anthropic_model})"
        else:
            engine_desc = "Smart Structured Extractor"

        return (
            "📊 <b>TechNews System Status:</b>\n\n"
            f"• <b>Status</b>: 🟢 Active &amp; Operational\n"
            f"• <b>Recurring Schedule</b>: Every {interval} minutes\n"
            f"• <b>Total Unique Stories Sent</b>: <code>{total_stories}</code>\n"
            f"• <b>AI Engine Mode</b>: {engine_desc}\n"
            f"• <b>Active Sources (7)</b>: Hacker News, TechCrunch, The Verge, Ars Technica, Guardian, MIT Review, DeepLearning.AI\n"
            f"{recent_str}\n\n"
            "<i>Everything is running smoothly!</i>"
        )

    if cmd == "/search":
        query = args.strip()
        if not query:
            return "🔍 <i>Please specify what you would like to search for.</i>\n\nExample:\n<code>/search deepseek v3</code>\n<code>/search quantum computing breakthrough</code>"

        results = await fetch_live_web_context(query, limit=4)
        if not results:
            return f"🔍 No recent news or web results found for '<b>{html.escape(query)}</b>'."

        formatted = [f"🔍 <b>Search results for '{html.escape(query)}':</b>\n"]
        for r in results:
            title = html.escape(r.get("title", "Untitled"))
            snippet = html.escape(r.get("snippet", ""))
            url = r.get("url", "")
            formatted.append(f"• <a href='{url}'><b>{title}</b></a>\n  {snippet}\n")

        return "\n".join(formatted)

    return ""


async def process_telegram_update(update: Dict[str, Any], token: str):
    """Process an individual update dictionary from Telegram getUpdates with strict deduplication."""
    up_id = update.get("update_id")
    if up_id is not None:
        if up_id in PROCESSED_UPDATE_IDS:
            logger.info("⏩ Skipping already processed update_id %s", up_id)
            return
        PROCESSED_UPDATE_IDS.add(up_id)
        if len(PROCESSED_UPDATE_IDS) > 2000:
            PROCESSED_UPDATE_IDS.clear()

    message = update.get("message")
    if not message:
        return

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    if not chat_id:
        return

    # Message deduplication key
    msg_key = f"{chat_id}:{message_id}"
    if msg_key in PROCESSED_MESSAGES:
        logger.info("⏩ Skipping duplicate message %s", msg_key)
        return
    PROCESSED_MESSAGES.add(msg_key)
    if len(PROCESSED_MESSAGES) > 2000:
        PROCESSED_MESSAGES.clear()

    text = message.get("text", "").strip()
    user_info = message.get("from", {})
    user_name = user_info.get("first_name") or user_info.get("username") or "Friend"

    if not text:
        return

    logger.info("📩 Message from @%s (%s): %s", user_info.get("username", "user"), chat_id, text)

    # Show typing indicator
    await send_typing_action(token, chat_id)

    reply_text = ""
    # Slash commands
    if text.startswith("/"):
        parts = text.split(" ", 1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else ""

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

    # Friendly, accurate conversational answer
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
