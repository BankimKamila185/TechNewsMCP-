"""
MCP Server for tech news aggregation, research (web/doc search), and Telegram messaging.

Tools:
1. fetch_tech_news(source, limit) -> JSON string
2. web_search(query, limit) -> JSON string
3. doc_search(url, max_chars) -> JSON string
4. send_telegram_message(text) -> str
"""

import asyncio
import json
import os
import re
import sys
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import feedparser
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load local environment variables from .env if present
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("tech-news-server")

# Known RSS Feeds
RSS_FEEDS: Dict[str, str] = {
    "techcrunch": "https://techcrunch.com/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
    "arstechnica": "https://feeds.arstechnica.com/arstechnica/index",
    "guardian": "https://www.theguardian.com/technology/rss",
    "mit": "https://www.technologyreview.com/feed/",
    "deeplearning_ai": "https://kill-the-newsletter.com/feeds/jc8116b7aauska20.xml",
}

HN_TOP_STORIES_ENDPOINT = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_ENDPOINT_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
TELEGRAM_API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
MAX_TELEGRAM_CHUNK_SIZE = 4000
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def split_telegram_message(
    text: str, max_chars: int = MAX_TELEGRAM_CHUNK_SIZE
) -> List[str]:
    """Split message text into chunks of at most max_chars on line boundaries where possible."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    lines = text.splitlines(keepends=True)
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        if len(line) > max_chars:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0
            start = 0
            while start < len(line):
                chunks.append(line[start : start + max_chars])
                start += max_chars
        elif current_len + len(line) > max_chars:
            chunks.append("".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


async def _fetch_hn_item(
    client: httpx.AsyncClient, item_id: int
) -> Optional[Dict[str, Any]]:
    """Fetch details for an individual Hacker News item."""
    try:
        url = HN_ITEM_ENDPOINT_TEMPLATE.format(item_id=item_id)
        resp = await client.get(url, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def _fetch_hackernews_stories(limit: int) -> List[Dict[str, Any]]:
    """Fetch top stories from Hacker News concurrently via Firebase API."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        resp = await client.get(HN_TOP_STORIES_ENDPOINT)
        if resp.status_code != 200:
            raise RuntimeError(f"Hacker News API returned status {resp.status_code}")
        story_ids = resp.json()[:limit]
        items = await asyncio.gather(*[_fetch_hn_item(client, sid) for sid in story_ids])

        results: List[Dict[str, Any]] = []
        for item in items:
            if not item:
                continue
            item_id = item.get("id")
            title = item.get("title", "Untitled").strip()
            url = item.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
            score = item.get("score", 0)
            results.append({
                "title": title,
                "url": url,
                "source": "hackernews",
                "score": score,
            })
        return results


def _parse_rss_content_sync(
    content_or_url: Any, source_name: str, limit: int
) -> List[Dict[str, Any]]:
    """Parse RSS feed synchronously using feedparser in a thread."""
    parsed = feedparser.parse(content_or_url)
    results: List[Dict[str, Any]] = []
    for entry in parsed.entries[:limit]:
        title = entry.get("title", "Untitled").strip()
        link = entry.get("link", "").strip()
        summary = entry.get("summary", "").strip()
        # Clean basic HTML tags from summary
        clean_summary = re.sub(r"<[^>]+>", "", summary).strip()
        results.append({
            "title": title,
            "url": link,
            "source": source_name,
            "summary": clean_summary,
        })
    return results


async def _fetch_rss_stories(source_name: str, feed_url: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch RSS feed via httpx with proper headers and parse via feedparser in a thread."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as client:
        resp = await client.get(feed_url)
        if resp.status_code != 200:
            raise RuntimeError(f"RSS feed '{source_name}' returned status {resp.status_code}")
        return await asyncio.to_thread(_parse_rss_content_sync, resp.content, source_name, limit)


@mcp.tool()
async def fetch_tech_news(source: str = "hackernews", limit: int = 10) -> str:
    """Fetch tech news from hackernews, techcrunch, verge, arstechnica, guardian, mit, or deeplearning_ai.

    Args:
        source: The news source name. Defaults to 'hackernews'.
        limit: Number of items to retrieve (clamped between 1 and 30). Defaults to 10.

    Returns:
        JSON string containing list of articles or an error object.
    """
    clamped_limit = max(1, min(30, int(limit)))
    norm_source = source.lower().strip()

    valid_sources = [
        "hackernews",
        "techcrunch",
        "verge",
        "arstechnica",
        "guardian",
        "mit",
        "deeplearning_ai",
    ]
    if norm_source not in valid_sources:
        return json.dumps({
            "error": f"Unknown source: '{source}'. Valid sources are: {', '.join(valid_sources)}"
        })

    try:
        if norm_source == "hackernews":
            articles = await _fetch_hackernews_stories(clamped_limit)
        else:
            feed_url = RSS_FEEDS[norm_source]
            articles = await _fetch_rss_stories(norm_source, feed_url, clamped_limit)
        return json.dumps(articles)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch news from '{source}': {str(e)}"})


@mcp.tool()
async def web_search(query: str, limit: int = 5) -> str:
    """Search the web for news research, background information, or technical context.

    Args:
        query: Search query text.
        limit: Number of search results to return (clamped between 1 and 10). Defaults to 5.

    Returns:
        JSON string containing list of search results ({title, url, snippet}).
    """
    clamped_limit = max(1, min(10, int(limit)))
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
            )
            if resp.status_code != 200:
                return json.dumps({"error": f"Search engine returned status {resp.status_code}"})

            results: List[Dict[str, str]] = []
            snippets = re.findall(
                r"<a[^>]+class=[\"']result__snippet[\"'][^>]*>(.*?)</a>",
                resp.text,
                re.DOTALL,
            )
            titles = re.findall(
                r"<a[^>]+class=[\"']result__a[\"'][^>]*href=[\"'](.*?)[\"'][^>]*>(.*?)</a>",
                resp.text,
                re.DOTALL,
            )

            for i in range(min(len(titles), clamped_limit)):
                raw_url, raw_title = titles[i]
                title = unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
                snippet = (
                    unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
                    if i < len(snippets)
                    else ""
                )
                # Unpack DuckDuckGo redirect url if applicable
                actual_url = raw_url
                if "uddg=" in actual_url:
                    parts = actual_url.split("uddg=")
                    if len(parts) > 1:
                        actual_url = unquote(parts[1].split("&")[0])

                results.append({"title": title, "url": actual_url, "snippet": snippet})

            return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}"})


@mcp.tool()
async def doc_search(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract readable plain text content from a web article or documentation page.

    Args:
        url: The web URL to inspect.
        max_chars: Maximum characters of readable text to extract (defaults to 3000).

    Returns:
        JSON string containing the extracted text and metadata.
    """
    clamped_chars = max(500, min(10000, int(max_chars)))
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return json.dumps({
                    "error": f"Failed to fetch document: HTTP {resp.status_code}"
                })

            raw_html = resp.text
            # Remove scripts, styles, and non-content tags
            cleaned = re.sub(
                r"<(script|style|noscript|header|footer|nav)[^>]*>.*?</\1>",
                " ",
                raw_html,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove all HTML tags
            text_only = re.sub(r"<[^>]+>", " ", cleaned)
            # Unescape HTML entities and collapse whitespace
            readable = unescape(re.sub(r"\s+", " ", text_only)).strip()

            content = readable[:clamped_chars]
            return json.dumps({
                "url": url,
                "character_count": len(content),
                "content": content,
            })
    except Exception as e:
        return json.dumps({"error": f"Document search failed: {str(e)}"})


@mcp.tool()
async def send_telegram_message(text: str) -> str:
    """Send an HTML-formatted message to Telegram, splitting into chunks if necessary.

    Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment."

    if not text or not text.strip():
        return "Error: Message text is empty."

    chunks = split_telegram_message(text, MAX_TELEGRAM_CHUNK_SIZE)
    url = TELEGRAM_API_URL_TEMPLATE.format(token=token)

    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, chunk in enumerate(chunks, start=1):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            try:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    safe_body = response.text.replace(token, "[REDACTED]")
                    return f"Telegram API error (status {response.status_code}): {safe_body}"
            except Exception as e:
                safe_err = str(e).replace(token, "[REDACTED]")
                return f"Telegram request failed on chunk {idx}/{len(chunks)}: {safe_err}"

    return f"Telegram message sent successfully ({len(chunks)} chunk(s))."


if __name__ == "__main__":
    mcp.run()
