"""
Unit tests for tech-news-mcp server tools and helper functions.
"""

import asyncio
import json
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from server import (
    MAX_TELEGRAM_CHUNK_SIZE,
    doc_search,
    fetch_tech_news,
    send_telegram_message,
    split_telegram_message,
    web_search,
)
from state_store import (
    are_titles_similar,
    filter_unsent_articles,
    get_sent_urls,
    init_db,
    mark_articles_as_sent,
    mark_urls_as_sent,
    normalize_url,
)


# ==============================================================================
# Chunk Splitting Tests
# ==============================================================================


def test_split_telegram_message_empty():
    """Empty string returns an empty list."""
    assert split_telegram_message("") == []


def test_split_telegram_message_short():
    """Text below max_chars returns a single chunk unchanged."""
    text = "Hello world! This is a short news update."
    chunks = split_telegram_message(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_telegram_message_on_newlines():
    """Text above max_chars splits cleanly on newline boundaries without cutting lines."""
    line_1 = "Story 1: Python 3.12 released.\n"
    line_2 = "Story 2: FastMCP makes building MCP servers easy.\n"
    line_3 = "Story 3: LangGraph enables stateful agent workflows.\n"
    full_text = line_1 + line_2 + line_3

    limit = len(line_1) + len(line_2) + 5
    chunks = split_telegram_message(full_text, max_chars=limit)

    assert len(chunks) == 2
    assert chunks[0] == line_1 + line_2
    assert chunks[1] == line_3
    assert "".join(chunks) == full_text
    assert all(len(c) <= limit for c in chunks)


def test_split_telegram_message_exceedingly_long_line():
    """A single unbroken line that exceeds max_chars is sliced cleanly into max_chars pieces."""
    long_line = "A" * 9500
    chunks = split_telegram_message(long_line, max_chars=4000)

    assert len(chunks) == 3
    assert len(chunks[0]) == 4000
    assert len(chunks[1]) == 4000
    assert len(chunks[2]) == 1500
    assert "".join(chunks) == long_line


def test_split_telegram_message_mixed_content():
    """Preserves integrity when long lines and normal lines are interspersed."""
    text = "Short header\n" + ("X" * 5000) + "\nShort footer\n"
    chunks = split_telegram_message(text, max_chars=4000)
    assert "".join(chunks) == text
    assert all(len(c) <= 4000 for c in chunks)


# ==============================================================================
# Unknown Source & Input Clamping Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_fetch_tech_news_unknown_source():
    """Requesting an unsupported source returns a JSON error."""
    result_str = await fetch_tech_news(source="myspace_tech", limit=5)
    data = json.loads(result_str)

    assert "error" in data
    assert "Unknown source" in data["error"]
    assert "myspace_tech" in data["error"]


@pytest.mark.asyncio
async def test_fetch_tech_news_limit_clamping():
    """Limit parameter is clamped to between 1 and 30."""
    with patch("server._fetch_hackernews_stories", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [{"title": "Test", "url": "https://example.com", "source": "hackernews"}]

        # Test limit < 1 clamps to 1
        await fetch_tech_news(source="hackernews", limit=-10)
        mock_fetch.assert_called_with(1)

        # Test limit > 30 clamps to 30
        await fetch_tech_news(source="hackernews", limit=100)
        mock_fetch.assert_called_with(30)


# ==============================================================================
# Mocked Network Tests: News Fetching & New Sources
# ==============================================================================


@pytest.mark.asyncio
async def test_fetch_hackernews_mocked():
    """Test HackerNews API parsing with mocked httpx responses."""
    fake_top_stories = [101, 102]
    fake_item_1 = {
        "id": 101,
        "title": "Quantum Computing Breakthrough",
        "url": "https://example.com/quantum",
        "score": 420,
    }
    fake_item_2 = {
        "id": 102,
        "title": "Ask HN: Favorite developer tools?",
        "score": 85,
    }

    async def mock_get(url, *args, **kwargs):
        if "topstories.json" in str(url):
            return httpx.Response(200, json=fake_top_stories)
        elif "101.json" in str(url):
            return httpx.Response(200, json=fake_item_1)
        elif "102.json" in str(url):
            return httpx.Response(200, json=fake_item_2)
        return httpx.Response(404)

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result_str = await fetch_tech_news(source="hackernews", limit=2)
        items = json.loads(result_str)

        assert len(items) == 2
        assert items[0]["title"] == "Quantum Computing Breakthrough"
        assert items[0]["url"] == "https://example.com/quantum"
        assert items[0]["source"] == "hackernews"
        assert items[0]["score"] == 420

        assert items[1]["title"] == "Ask HN: Favorite developer tools?"
        assert items[1]["url"] == "https://news.ycombinator.com/item?id=102"
        assert items[1]["score"] == 85


@pytest.mark.asyncio
async def test_fetch_new_sources_mocked():
    """Verify Guardian, MIT, and DeepLearning.AI RSS parsing."""
    mock_rss_xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0">
      <channel>
        <title>MIT Tech Review</title>
        <item>
          <title>Frontiers in Artificial Intelligence</title>
          <link>https://technologyreview.com/2026/ai-frontiers</link>
          <description>Deep dive into foundation models</description>
        </item>
      </channel>
    </rss>
    """
    mock_resp = httpx.Response(200, content=mock_rss_xml)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        for src in ["guardian", "mit", "deeplearning_ai"]:
            result_str = await fetch_tech_news(source=src, limit=3)
            items = json.loads(result_str)
            assert len(items) == 1
            assert items[0]["title"] == "Frontiers in Artificial Intelligence"
            assert items[0]["source"] == src


# ==============================================================================
# Research Tools: web_search and doc_search
# ==============================================================================


@pytest.mark.asyncio
async def test_web_search_mocked():
    """Test web_search tool with mocked search engine results."""
    mock_html = """
    <html>
      <a class="result__a" href="https://example.com/research">Claude 3.7 Sonnet Released</a>
      <a class="result__snippet">Anthropic announced a new hybrid reasoning model.</a>
    </html>
    """
    mock_resp = httpx.Response(200, text=mock_html)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await web_search("claude hybrid reasoning", limit=2)
        data = json.loads(res)
        assert len(data) == 1
        assert data[0]["title"] == "Claude 3.7 Sonnet Released"
        assert "hybrid reasoning" in data[0]["snippet"]


@pytest.mark.asyncio
async def test_doc_search_mocked():
    """Test doc_search tool extracts clean text from HTML."""
    html_page = """
    <html>
      <head><script>alert('hidden');</script></head>
      <body>
        <h1>Deep Learning Breakthrough</h1>
        <p>Researchers have introduced a scalable transformer architecture.</p>
      </body>
    </html>
    """
    mock_resp = httpx.Response(200, text=html_page)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await doc_search("https://example.com/article", max_chars=1000)
        data = json.loads(res)
        assert data["url"] == "https://example.com/article"
        assert "Deep Learning Breakthrough" in data["content"]
        assert "alert" not in data["content"]


# ==============================================================================
# Mocked Telegram Tests & Secret Redaction
# ==============================================================================


@pytest.mark.asyncio
async def test_send_telegram_missing_env(monkeypatch):
    """Missing bot token or chat ID returns a descriptive error without crashing."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    res = await send_telegram_message("Hello")
    assert "Error" in res
    assert "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set" in res


@pytest.mark.asyncio
async def test_send_telegram_success_mocked(monkeypatch):
    """Successful Telegram send returns success status."""
    fake_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", fake_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987654321")

    mock_resp = httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await send_telegram_message("<b>News digest</b>")
        assert "successfully" in res


@pytest.mark.asyncio
async def test_send_telegram_error_redaction_mocked(monkeypatch):
    """Non-200 responses return an error and NEVER leak the bot token."""
    secret_token = "TOP_SECRET_BOT_TOKEN_XYZ987"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    error_body = f'{{"ok":false,"error_code":401,"description":"Invalid token {secret_token}"}}'
    mock_resp = httpx.Response(401, text=error_body)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await send_telegram_message("Test message")

        assert "Telegram API error (status 401)" in res
        assert secret_token not in res
        assert "[REDACTED]" in res


# ==============================================================================
# State Store (sent.db) Tests
# ==============================================================================


def test_state_store_operations(tmp_path):
    """Test state_store SQLite helper for deduplication and insertion."""
    test_db = str(tmp_path / "test_sent.db")

    init_db(test_db)
    assert get_sent_urls(test_db) == set()

    articles = [
        {"title": "Article 1", "url": "https://example.com/1"},
        {"title": "Article 2", "url": "https://example.com/2"},
        {"title": "Article 1 Dupe", "url": "https://example.com/1"},
    ]

    fresh = filter_unsent_articles(articles, db_path=test_db)
    assert len(fresh) == 2
    assert {a["url"] for a in fresh} == {"https://example.com/1", "https://example.com/2"}

    mark_urls_as_sent(["https://example.com/1"], db_path=test_db)
    assert get_sent_urls(test_db) == {"https://example.com/1"}

    fresh_after = filter_unsent_articles(articles, db_path=test_db)
    assert len(fresh_after) == 1
    assert fresh_after[0]["url"] == "https://example.com/2"

    mark_urls_as_sent(["https://example.com/1", "https://example.com/2"], db_path=test_db)
    assert len(get_sent_urls(test_db)) >= 2


def test_url_normalization():
    """Verify that tracking query parameters, www prefixes, and trailing slashes are stripped."""
    u1 = "https://www.theverge.com/2026/10/28/story/?utm_source=rss&utm_medium=feed&guccounter=1#comments"
    u2 = "https://theverge.com/2026/10/28/story"
    assert normalize_url(u1) == normalize_url(u2)


def test_semantic_title_similarity():
    """Verify that headlines about the same event from different outlets are identified as duplicates."""
    # Cross-outlet duplicate examples
    assert are_titles_similar(
        "Upwind raises $400M after walking away from Datadog acquisition offer",
        "Upwind secures $400M valuation after rejecting Datadog acquisition bid",
    )
    assert are_titles_similar(
        "Anduril now says its BlueChip laptop will meet Buy American standards at launch",
        "Anduril and Qualcomm partner to build Made-in-USA laptop",
    )
    # Distinct topics should not match
    assert not are_titles_similar(
        "Apple announces new M4 MacBook Pro with faster Neural Engine",
        "Google launches Gemini 2.0 with real-time video features",
    )


def test_cross_publication_and_tracking_deduplication(tmp_path):
    """Verify that filter_unsent_articles drops both tracking variations and cross-outlet duplicates."""
    test_db = str(tmp_path / "dedupe_test.db")
    init_db(test_db)

    batch_1 = [
        {
            "title": "Anduril now says its BlueChip laptop will meet Buy American standards",
            "url": "https://www.theverge.com/policy/anduril-bluechip?utm_source=rss",
        },
        # Same event reported by TechCrunch in the same batch
        {
            "title": "Anduril and Qualcomm partner to build Made-in-USA laptop",
            "url": "https://techcrunch.com/2026/anduril-laptop-qualcomm",
        },
        # Different story
        {
            "title": "ULA Vulcan rocket returns to flight on Friday",
            "url": "https://arstechnica.com/space/vulcan-flight",
        },
    ]

    fresh = filter_unsent_articles(batch_1, db_path=test_db)
    # The duplicate Anduril story should be collapsed, keeping only 2 distinct stories
    assert len(fresh) == 2
    titles = [a["title"] for a in fresh]
    assert any("Vulcan" in t for t in titles)
    assert any("Anduril" in t for t in titles)

    # Mark these as sent
    mark_articles_as_sent(fresh, db_path=test_db)

    # Next cycle: Ars Technica publishes Anduril story with new URL, and a new story appears
    batch_2 = [
        {
            "title": "Anduril announces US-built laptops with Qualcomm chips",
            "url": "https://arstechnica.com/gadgets/anduril-us-laptop",
        },
        {
            "title": "DeepMind announces major protein folding breakthrough",
            "url": "https://deeplearning.ai/protein-breakthrough",
        },
    ]

    fresh_2 = filter_unsent_articles(batch_2, db_path=test_db)
    # The Anduril story must be dropped because it was already sent in batch 1!
    assert len(fresh_2) == 1
    assert "DeepMind" in fresh_2[0]["title"]

