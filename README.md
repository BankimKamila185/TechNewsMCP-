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
  - `send_telegram_photo(caption, photo_url)`: Delivers high-res featured story images and infographics directly to Telegram.
- **LangGraph Multi-Step Workflow (`graph.py`)**:
  - Node 1: `fetch_news` - Concurrently pulls candidate stories across all sources (extracting article thumbnails and media content).
  - Node 2: `filter_new` - Compares candidate URLs against SQLite (`sent.db`). Bypasses remaining steps if no new stories exist.
  - Node 3: `research` - Runs deep-dive research queries via MCP tools (`web_search` and `doc_search`) on the top candidate stories.
  - Node 4: `summarize` - Formats the digest into structured topic categories:
    - 🤖 **New Models & AI** (LLMs, neural networks, foundation model releases, AI research)
    - 💡 **New Innovation** (Scientific breakthroughs, algorithms, quantum computing, clean tech)
    - 🛡️ **Cyber Crime & Security** (Breaches, zero-days, exploits, malware, CVEs)
    - ⚡ **New Tech & Hardware** (Chips, robotics, space tech, devices, semiconductors)
    - 🌐 **Industry & Big Tech** (Major funding, regulations, acquisitions)
    - Uses Claude (`claude-sonnet-4-6`) when `ANTHROPIC_API_KEY` is present, or a smart categorized extractive fallback.
  - Node 5: `send` - Delivers the categorized digest with the featured headline image to Telegram, and commits URLs to `sent.db`.
- **24/7 Interactive Telegram Chatbot (`telegram_bot.py`)**:
  - Chat directly with the bot on Telegram! It discusses technology, AI advancements, semiconductors, cybersecurity, and recent tech trends.
  - Powered by Claude (`ChatAnthropic`) when an API key is provided, or an intelligent contextual tech fallback assistant.
  - **Slash Commands**:
    - `<b>/start</b>` or `<b>/help</b>`: Bot introduction, quick tips, and available features.
    - `<b>/news</b>` or `<b>/digest</b>`: Instantly fetch and send fresh breaking news right now.
    - `<b>/search &lt;query&gt;</b>`: Real-time web and tech archive search on any topic (e.g. `/search deepseek`).
    - `<b>/status</b>`: Check system uptime, total sent stories, and scheduler interval.
    - `<b>/categories</b>`: View description of the 5 tracked tech domains.
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
├── app.py                    # Web server entrypoint for Render (health check + background scheduler & chat)
├── render.yaml               # Render Blueprint configuration for one-click deployment
├── server.py                 # FastMCP server (fetch_tech_news, web_search, doc_search, send_telegram_message)
├── graph.py                  # LangGraph agent workflow (fetch -> filter -> research -> summarize -> send)
├── scheduler.py              # Background recurring runner (hourly news digests + real-time chat listener)
├── telegram_bot.py           # Interactive Telegram chatbot engine (chat replies, slash commands, search)
├── state_store.py            # SQLite helper for sent-URL persistence & deduplication
├── connect_telegram.py       # Helper script to link your Telegram bot and auto-detect Chat ID
├── requirements.txt          # Python dependencies (MCP, LangGraph, Uvicorn, Starlette, etc.)
├── pytest.ini                # Pytest configuration
├── .env.example              # Sample environment variables template
├── .gitignore                # Git ignore rules (.env, .venv, *.db, __pycache__)
├── README.md                 # Complete documentation & setup guide
├── tests/
│   ├── test_app.py           # Pytest test suite for Render web endpoints
│   ├── test_server.py        # Pytest test suite for server tools & chunking
│   ├── test_graph.py         # Pytest test suite for graph routing & sanitization
│   └── test_telegram_bot.py  # Pytest test suite for Telegram chatbot & command responses
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

## ⏱️ Running Automatically Every 1 Hour

### Option 1: Run Local Scheduler Daemon (Recommended for local dev)
Use the included `scheduler.py` script:
```bash
# Runs the pipeline every 1 hour continuously (default is 60 minutes)
python scheduler.py --hours 1
```
- It performs an initial check immediately on launch.
- Every 1 hour, it queries all 7 tech publications, filters out already-sent articles using `sent.db`, runs research on fresh stories, and sends them with featured images to your Telegram.
- If no new stories were published during that hour, it outputs `Status: No new stories` without spamming your chat.
- Stop anytime with `Ctrl + C`.

### Option 2: Run in Background with nohup
```bash
nohup .venv/bin/python scheduler.py --hours 1 > scheduler.log 2>&1 &
```

### Option 3: Deploy to Render (Cloud Hosting)
Deploy to Render as a 24/7 web service with zero manual server management. See [Deploying to Render](#-deploying-to-render) below for step-by-step instructions.

### Option 4: GitHub Actions (Cloud Automation)
The included workflow `.github/workflows/daily.yml` is configured with:
```yaml
on:
  schedule:
    - cron: '0 * * * *'
```
It runs on GitHub's cloud runners every 1 hour, using `actions/cache@v4` to persist `sent.db`.

---

## 🌐 Deploying to Render

You can host **TechNewsMCP** on [Render](https://render.com) for 24/7 cloud execution using either the **Blueprint** (infrastructure as code via `render.yaml`) or manual web service creation.

The project includes `app.py`, a lightweight web service powered by **Starlette** and **Uvicorn** that binds to Render's `$PORT`, responds to health checks, runs the hourly news digest in the background, and provides on-demand triggering via HTTP.

### Method 1: Blueprint Deployment (Recommended & Fastest)

1. **Log in** to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** (top right) and select **Blueprint**.
3. Connect your GitHub repository:
   ```
   https://github.com/BankimKamila185/TechNewsMCP-.git
   ```
4. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Service Type**: Web Service
   - **Runtime**: Python 3.12
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Fill in the prompted environment variables:
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`) for lightning-fast, highly accurate AI chat and summaries
   - `GROQ_MODEL`: `qwen/qwen3.8-27b`
   - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API token (e.g. `8548609073:AAG-mIOVCh-8ZHCPphkUvYRY3GztkNcdp8Q`)
   - `TELEGRAM_CHAT_ID`: Your chat ID (e.g. `7847910724`)
   - `CHECK_INTERVAL_MINUTES`: `60` (or `10` if you want 10-minute digests)
   - `ANTHROPIC_API_KEY`: *(Optional alternative LLM)*
6. Click **Apply Blueprint**. Render will build and deploy your service!

---

### Method 2: Manual Web Service Setup

If you prefer setting up manually without a Blueprint:

1. In the Render Dashboard, click **New +** -> **Web Service**.
2. Select your repository: `BankimKamila185/TechNewsMCP-`.
3. Configure the following fields:
   - **Name**: `tech-news-mcp`
   - **Region**: Any (e.g., *Oregon (US West)* or *Frankfurt (EU)*)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: `Free`
4. In the **Environment Variables** section, add:
   - `PYTHON_VERSION` = `3.12.0`
   - `TELEGRAM_BOT_TOKEN` = `8548609073:AAG-mIOVCh-8ZHCPphkUvYRY3GztkNcdp8Q`
   - `TELEGRAM_CHAT_ID` = `7847910724`
   - `CHECK_INTERVAL_MINUTES` = `60`
   - `ANTHROPIC_API_KEY` = *(Optional sk-ant-...)*
5. Click **Create Web Service**.

---

### 🌐 Endpoints & Testing Your Render Deployment

Once deployed, Render provides a public URL (e.g., `https://tech-news-mcp.onrender.com`):

- **Health Check**:
  ```bash
  curl https://your-app.onrender.com/health
  # Returns: {"status": "healthy", "service": "tech-news-mcp", "interval_minutes": 60}
  ```
- **Manual On-Demand Digest Trigger**:
  ```bash
  curl https://your-app.onrender.com/trigger
  # Triggers an immediate news fetch, categorization, and sends digest to Telegram!
  ```
- **Automatic Background Scheduling**:
  `app.py` runs an async background loop that automatically checks and sends fresh news to Telegram every 1 hour (or whichever interval you specify in `CHECK_INTERVAL_MINUTES`).
- **Sleep Prevention (Optional for Free Tier)**:
  Render free web services spin down after 15 minutes of inactivity. You can keep the service awake or trigger digests by setting up a free monitor (like [UptimeRobot](https://uptimerobot.com) or [cron-job.org](https://cron-job.org)) to ping `https://your-app.onrender.com/health` every 10–14 minutes, or trigger `https://your-app.onrender.com/trigger` every hour!

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
| `GROQ_API_KEY` | Groq API Key (`gsk_...`) for ultra-fast reasoning & summaries |
| `GROQ_MODEL` *(Optional)* | Model ID (defaults to `qwen/qwen3.8-27b`) |
| `ANTHROPIC_API_KEY` *(Optional)* | Anthropic Claude API Key (`sk-ant-...`) |
| `TELEGRAM_BOT_TOKEN` | `8548609073:AAG-mIOVCh-8ZHCPphkUvYRY3GztkNcdp8Q` |
| `TELEGRAM_CHAT_ID` | Your Telegram Chat ID (detected by `connect_telegram.py`) |
| `ANTHROPIC_MODEL` *(Optional)* | Claude Model ID (defaults to `claude-sonnet-4-6`) |
