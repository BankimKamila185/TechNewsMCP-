# 📰 tech-news-mcp

An automated daily tech news aggregation and research digest pipeline powered by a custom **Model Context Protocol (MCP)** server, a stateful **LangGraph** workflow, **Claude** (Anthropic), and **Telegram**.

```
                           ┌────────────────────────┐
                           │   LangGraph Agent      │
                           │      (graph.py)        │
                           └───────────┬────────────┘
                                       │ Stdio MCP
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastMCP Server (server.py)                      │
│                                                                        │
│  ┌───────────────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │    fetch_tech_news    │  │ web_search │  │  send_telegram_message │ │
│  └───────────┬───────────┘  │ doc_search │  └────────────┬───────────┘ │
│              │              └────────────┘               │             │
└──────────────┼───────────────────────────────────────────┼─────────────┘
               │                                           │
       ┌───────┴───────────────────────┐                   │
       ▼                               ▼                   ▼
┌──────────────┐      ┌───────────────────────────┐ ┌────────────────┐
│ Hacker News  │      │ Premium Tech Publications │ │ Telegram API   │
│ Firebase API │      │ • TechCrunch              │ │ HTML Formatted │
│ Concurrent   │      │ • The Verge               │ │ 4000-char Chunk│
│ asyncio      │      │ • Ars Technica            │ └────────────────┘
└──────────────┘      │ • The Guardian Tech       │
                      │ • MIT Technology Review   │
                      │ • DeepLearning.AI Batch   │
                      └───────────────────────────┘
```

---

## 🚀 Features

- **Custom FastMCP Server (`server.py`)**:
  - `fetch_tech_news(source, limit)`: Supports 7 major tech & AI sources:
    - **Hacker News** (official Firebase API, concurrent `asyncio.gather` fetching)
    - **TechCrunch** (RSS via `feedparser` in worker thread)
    - **The Verge** (RSS via `feedparser`)
    - **Ars Technica** (RSS via `feedparser`)
    - **The Guardian Technology** (RSS via `feedparser`)
    - **MIT Technology Review** (RSS via `feedparser`)
    - **DeepLearning.AI The Batch** (AI research feed)
  - `web_search(query, limit)`: Searches the web for background research and technical context on breaking stories.
  - `doc_search(url, max_chars)`: Extracts clean, readable text from target URLs/documentation for deep-dive analysis.
  - `send_telegram_message(text)`: Line-boundary chunking (< 4,000 characters), HTML parse mode, disabled web page previews, and strict token sanitization (secrets are never logged or leaked).
- **LangGraph Multi-Step Workflow (`graph.py`)**:
  - Node 1: `fetch_news` - Concurrently pulls candidate stories across all sources.
  - Node 2: `filter_new` - Compares candidate URLs against SQLite (`sent.db`). Bypasses remaining steps if no new stories exist.
  - Node 3: `research` - Runs deep-dive research queries via MCP tools (`web_search` and `doc_search`) on the top candidate stories.
  - Node 4: `summarize` - Uses Claude (`claude-sonnet-4-6`) to curate the top 8 stories, explaining why each matters with research context in Telegram HTML. (Includes an extractive curation fallback if `ANTHROPIC_API_KEY` is not yet configured).
  - Node 5: `send` - Delivers the digest to Telegram with header `📰 Tech News Digest` and commits URLs to `sent.db` only upon confirmed delivery.
- **One-Click Telegram Connection (`connect_telegram.py`)**:
  - Automatically captures your Chat ID once you click Start on your bot, updates `.env`, and sends a verification message.
- **Automated Scheduling (`.github/workflows/daily.yml`)**:
  - Triggers every morning at **02:30 UTC** (**08:00 AM IST**).
  - Also triggerable manually via `workflow_dispatch`.
  - Caches and restores `sent.db` using `actions/cache@v4`.

---

## 📁 Project Structure

```
tech-news-mcp/
├── server.py                 # FastMCP server (fetch_tech_news, web_search, doc_search, send_telegram_message)
├── graph.py                  # LangGraph agent workflow (fetch -> filter -> research -> summarize -> send)
├── state_store.py            # SQLite helper for sent-URL persistence & deduplication
├── connect_telegram.py       # Helper script to link your Telegram bot and auto-detect Chat ID
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── .env.example              # Sample environment variables template
├── .gitignore                # Git ignore rules (.env, .venv, *.db, __pycache__)
├── README.md                 # Complete documentation & setup guide
├── tests/
│   ├── test_server.py        # Pytest test suite for server tools & chunking
│   └── test_graph.py         # Pytest test suite for graph routing & sanitization
└── .github/
    └── workflows/
        └── daily.yml         # Daily scheduled GitHub Actions workflow (02:30 UTC / 08:00 AM IST)
```

---

## 🤖 1-Minute Telegram Bot Setup

Your bot is already registered on Telegram:
- **Bot Username**: [`@daily_techdigest_bot`](https://t.me/daily_techdigest_bot)
- **Bot Token**: Pre-configured in `.env`

### Connecting Your Account:
1. Open Telegram on your phone or desktop.
2. Search for **`@daily_techdigest_bot`** or open the link:
   👉 **[https://t.me/daily_techdigest_bot](https://t.me/daily_techdigest_bot)**
3. Click **START** (or send `/start` or `hi` to the bot).
4. In your terminal, run:
   ```bash
   python connect_telegram.py
   ```
   *The script will detect your message, save your `TELEGRAM_CHAT_ID` to `.env`, and send a confirmation message to your Telegram!*

---

## ⏱️ Running Automatically Every 10 Minutes

### Option 1: Run Local Scheduler Daemon (Recommended for local dev)
Use the included `scheduler.py` script:
```bash
# Runs the pipeline every 10 minutes continuously
python scheduler.py --interval 10
```
- It performs an initial check immediately on launch.
- Every 10 minutes, it queries all 7 tech publications, filters out already-sent articles using `sent.db`, runs research on fresh stories, and sends them to your Telegram.
- If no new stories were published during those 10 minutes, it outputs `Status: No new stories` without spamming your chat.
- Stop anytime with `Ctrl + C`.

### Option 2: Run in Background with nohup
```bash
nohup .venv/bin/python scheduler.py --interval 10 > scheduler.log 2>&1 &
```

### Option 3: GitHub Actions (Cloud Automation)
The included workflow `.github/workflows/daily.yml` is configured with:
```yaml
on:
  schedule:
    - cron: '*/10 * * * *'
```
It runs on GitHub's cloud runners every 10 minutes, using `actions/cache@v4` to persist `sent.db`.

---

## 💻 Local Installation & Running

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Run the LangGraph news agent
python graph.py
```

### Expected Output:
```text
Status: Digest sent successfully (8 stories sent and recorded)
```
Check your Telegram app: you will receive the full HTML formatted news digest!

---

## 🧪 Testing

Run the full automated test suite with `pytest`:

```bash
pytest -v
```

This tests:
- Telegram chunk splitting on newline boundaries and long-line slicing.
- Unknown news source handling and input limit clamping (1–30).
- Mocked Hacker News Firebase API fetching and Ask HN fallback URLs.
- Mocked RSS feeds for TechCrunch, The Verge, Ars Technica, Guardian, MIT Tech Review, and DeepLearning.AI.
- Mocked `web_search` and `doc_search` research tools.
- Telegram API success, failure, and strict secret redaction.
- SQLite `sent.db` deduplication logic.
- LangGraph conditional routing and Telegram HTML ampersand sanitization.

---

## 🔍 Testing the MCP Server Standalone

You can test the MCP server in isolation using the official MCP Developer Inspector:

```bash
mcp dev server.py
```

Or using `npx`:

```bash
npx @modelcontextprotocol/inspector python server.py
```

---

## ⚙️ GitHub Actions Daily Scheduling

The repository includes `.github/workflows/daily.yml` configured to run automatically:
- **Schedule**: Every day at **02:30 UTC** (**8:00 AM IST**).
- **Manual Trigger**: Can be run on demand from the **Actions** tab on GitHub via "Run workflow".

### Configuring GitHub Secrets
In your GitHub repository, navigate to **Settings > Secrets and variables > Actions** and add:

| Secret Name | Description |
| :--- | :--- |
| `ANTHROPIC_API_KEY` | Anthropic Claude API Key (`sk-ant-...`) |
| `TELEGRAM_BOT_TOKEN` | `8548609073:AAG-mIOVCh-8ZHCPphkUvYRY3GztkNcdp8Q` |
| `TELEGRAM_CHAT_ID` | Your Telegram Chat ID (detected by `connect_telegram.py`) |
| `ANTHROPIC_MODEL` *(Optional)* | Claude Model ID (defaults to `claude-sonnet-4-6`) |
