"""
LangGraph agent for aggregating tech news, performing deep research, summarizing with Claude, and sending to Telegram.
"""

import asyncio
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from state_store import (
    are_titles_similar,
    filter_unsent_articles,
    mark_articles_as_sent,
    normalize_url,
)

# Load environment variables
load_dotenv()

SERVER_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))

MCP_CONNECTIONS: Dict[str, Any] = {
    "tech_news_server": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [SERVER_SCRIPT_PATH],
    }
}

NEWS_SOURCES = [
    "hackernews",
    "techcrunch",
    "verge",
    "guardian",
    "mit",
    "deeplearning_ai",
]


class AgentState(TypedDict):
    news: List[Dict[str, Any]]
    fresh_news: List[Dict[str, Any]]
    research_notes: Dict[str, str]
    digest: str
    status: str


def extract_mcp_text(result: Any) -> str:
    """Extract string content from MCP tool return which can be a string or list of content blocks."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: List[str] = []
        for block in result:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text")))
            else:
                parts.append(str(block))
        return "".join(parts)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def sanitize_telegram_html(text: str) -> str:
    """Ensure raw ampersands are escaped as &amp; without double-escaping valid entities."""
    return re.sub(r"&(?!(amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)", "&amp;", text)


async def fetch_news(state: AgentState) -> Dict[str, Any]:
    """Node 1: Concurrently fetch news across all configured sources via MCP."""
    client = MultiServerMCPClient(MCP_CONNECTIONS)
    try:
        tools = await client.get_tools()
        fetch_tool = next((t for t in tools if t.name == "fetch_tech_news"), None)
        if not fetch_tool:
            raise RuntimeError("Tool 'fetch_tech_news' not exposed by MCP server.")

        all_news: List[Dict[str, Any]] = []
        seen_urls = set()

        for source in NEWS_SOURCES:
            try:
                res = await fetch_tool.ainvoke({"source": source, "limit": 6})
                text = extract_mcp_text(res)
                data = json.loads(text)
                if isinstance(data, list):
                    for item in data:
                        url = item.get("url")
                        norm_u = normalize_url(url)
                        if norm_u and norm_u not in seen_urls:
                            seen_urls.add(norm_u)
                            all_news.append(item)
                elif isinstance(data, dict) and "error" in data:
                    print(f"Warning fetching source '{source}': {data['error']}", file=sys.stderr)
            except Exception as src_err:
                print(f"Error fetching from {source}: {src_err}", file=sys.stderr)

        if not all_news:
            raise RuntimeError("No stories could be fetched from any news source.")

        return {
            "news": all_news,
            "status": f"Fetched {len(all_news)} stories across sources",
        }
    except Exception as e:
        err_msg = f"Failed to fetch news: {str(e)}"
        print(f"Fetch news error: {err_msg}", file=sys.stderr)
        # Attempt to notify Telegram on fetch failure if Telegram tool is accessible
        try:
            tools = await client.get_tools()
            send_tool = next((t for t in tools if t.name == "send_telegram_message"), None)
            if send_tool:
                await send_tool.ainvoke({
                    "text": f"⚠️ <b>Tech News Digest Error</b>\n{sanitize_telegram_html(err_msg)}"
                })
        except Exception:
            pass
        return {
            "news": [],
            "fresh_news": [],
            "research_notes": {},
            "digest": "",
            "status": f"Fetch failed: {str(e)}",
        }


async def filter_new(state: AgentState) -> Dict[str, Any]:
    """Node 2: Drop URLs already recorded in sent.db using state_store.py."""
    if state.get("status", "").startswith("Fetch failed"):
        return {"fresh_news": [], "status": state.get("status", "Fetch failed")}

    current_news = state.get("news", [])
    fresh = filter_unsent_articles(current_news)

    if not fresh:
        return {"fresh_news": [], "research_notes": {}, "status": "No new stories"}

    return {
        "fresh_news": fresh,
        "status": f"Found {len(fresh)} new stories to research and summarize",
    }


def route_after_filter(state: AgentState) -> str:
    """Conditional edge after filter_new: if no fresh news, skip directly to END."""
    fresh = state.get("fresh_news", [])
    if not fresh:
        return "end"
    return "research"


async def research_stories(state: AgentState) -> Dict[str, Any]:
    """Node 3: Perform research using MCP doc_search and web_search for top candidates."""
    fresh = state.get("fresh_news", [])
    if not fresh:
        return {"research_notes": {}, "status": "No stories to research"}

    client = MultiServerMCPClient(MCP_CONNECTIONS)
    notes: Dict[str, str] = {}

    try:
        tools = await client.get_tools()
        doc_tool = next((t for t in tools if t.name == "doc_search"), None)
        web_tool = next((t for t in tools if t.name == "web_search"), None)

        # Inspect top 6 candidate stories for deeper research
        candidates = fresh[:6]
        for item in candidates:
            url = item.get("url")
            title = item.get("title", "")
            if not url:
                continue

            # If the item already has a good summary (from RSS), use it
            if item.get("summary") and len(item["summary"]) > 100:
                notes[url] = item["summary"][:350]
                continue

            # Otherwise, use web_search or doc_search to research context
            if web_tool and title:
                try:
                    res = await web_tool.ainvoke({"query": f"{title} tech news", "limit": 2})
                    parsed_results = json.loads(extract_mcp_text(res))
                    if isinstance(parsed_results, list) and parsed_results:
                        snippets = [r.get("snippet", "") for r in parsed_results if r.get("snippet")]
                        if snippets:
                            notes[url] = " ".join(snippets)[:350]
                except Exception:
                    pass

    except Exception as e:
        print(f"Research node warning: {e}", file=sys.stderr)

    return {
        "research_notes": notes,
        "status": f"Completed research on {len(notes)} stories",
    }


SUMMARIZE_SYSTEM_PROMPT = """You are an expert tech journalist and research analyst compiling a daily digest for Telegram.
Review the provided candidate tech stories and research notes, and select the most significant, impactful ones (at most 8 stories).

STRICT DEDUPLICATION RULES:
- Every story in the digest MUST be about a completely distinct event or topic.
- NEVER include multiple stories about the same company announcement, event, launch, acquisition, or incident, even if reported by different outlets.
- If multiple candidates cover the same story (e.g. from TechCrunch and The Verge), pick only the single best, most informative version.

Formatting Rules:
1. Output ONLY valid Telegram HTML using ONLY <b> and <a href="..."> tags.
2. Format each story exactly like this on its own paragraph:
• <a href="{url}"><b>{escaped_title}</b></a> - {one plain sentence explaining why this story matters based on the research.}
3. Do NOT include markdown formatting (no asterisks, no markdown links, no code fences like ```html).
4. Do NOT include introductory remarks, commentary, greetings, or conclusions.
5. All '&' characters in text and URLs MUST be written as '&amp;'.
6. Do NOT include more than 8 stories.
"""

SUMMARIZE_USER_PROMPT = """Candidate stories and research notes:
{candidate_stories}

Generate the Telegram HTML digest now:"""


async def summarize(state: AgentState) -> Dict[str, Any]:
    """Node 4: Summarize fresh news using ChatAnthropic (with extractive research fallback if API key is not configured)."""
    fresh_news = state.get("fresh_news", [])
    research_notes = state.get("research_notes", {})
    if not fresh_news:
        return {"digest": "", "status": "No new stories"}

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # Fallback if ANTHROPIC_API_KEY is not yet populated
    if not api_key or api_key.startswith("your_anthropic_api_key"):
        # Generate high-quality curated digest with zero duplicate stories
        digest_lines = []
        selected_titles = []
        for item in fresh_news:
            title = item.get("title", "Tech News Update").strip()
            # Double-check title is not duplicate of an already chosen story in this digest
            if any(are_titles_similar(title, st) for st in selected_titles):
                continue
            selected_titles.append(title)

            clean_title = sanitize_telegram_html(title)
            url = item.get("url", "#")
            research_info = research_notes.get(url) or item.get("summary") or "Key development impacting the industry."
            sentence = re.split(r"(?<=[.!?])\s+", research_info.strip())[0] if research_info else "Major industry update."
            clean_sentence = sanitize_telegram_html(sentence)
            digest_lines.append(f'• <a href="{url}"><b>{clean_title}</b></a> - {clean_sentence}')
            if len(digest_lines) >= 8:
                break

        clean_digest = "\n\n".join(digest_lines)
        return {"digest": clean_digest, "status": "Summarization complete (curated digest)"}

    # If ANTHROPIC_API_KEY is available, use Claude
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    lines = []
    for idx, item in enumerate(fresh_news[:25], 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        source = item.get("source", "")
        extra = f" (HN Score: {item['score']})" if "score" in item else ""
        context = research_notes.get(url) or item.get("summary") or ""
        if context:
            extra += f" | Research context: {context[:200]}"
        lines.append(f"{idx}. [{source.upper()}] {title}{extra}\n   URL: {url}")
    candidate_text = "\n".join(lines)

    try:
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.2,
            max_tokens=2000,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZE_SYSTEM_PROMPT),
            ("user", SUMMARIZE_USER_PROMPT),
        ])
        chain = prompt | llm
        response = await chain.ainvoke({"candidate_stories": candidate_text})
        raw_digest = response.content if isinstance(response.content, str) else str(response.content)

        # Strip accidental markdown code blocks
        clean_digest = raw_digest.strip()
        if clean_digest.startswith("```html"):
            clean_digest = clean_digest[7:]
        elif clean_digest.startswith("```"):
            clean_digest = clean_digest[3:]
        if clean_digest.endswith("```"):
            clean_digest = clean_digest[:-3]
        clean_digest = clean_digest.strip()

        clean_digest = sanitize_telegram_html(clean_digest)

        return {"digest": clean_digest, "status": "Summarization complete"}
    except Exception as e:
        err_msg = f"Summarization failed: {str(e)}"
        print(err_msg, file=sys.stderr)
        return {"digest": "", "status": err_msg}


async def send(state: AgentState) -> Dict[str, Any]:
    """Node 5: Send digest to Telegram and record sent URLs upon success."""
    digest = state.get("digest", "").strip()
    if not digest:
        return {"status": state.get("status", "No digest to send")}

    header = "📰 <b>Tech News Digest</b>\n\n"
    full_message = f"{header}{digest}"

    client = MultiServerMCPClient(MCP_CONNECTIONS)
    try:
        tools = await client.get_tools()
        send_tool = next((t for t in tools if t.name == "send_telegram_message"), None)
        if not send_tool:
            raise RuntimeError("Tool 'send_telegram_message' not exposed by MCP server.")

        res = await send_tool.ainvoke({"text": full_message})
        res_text = extract_mcp_text(res)

        if "error" in res_text.lower() or "failed" in res_text.lower():
            return {"status": f"Failed to send digest: {res_text}"}

        # Only after a successful send, record articles with canonical URLs and titles in sent.db
        fresh = state.get("fresh_news", [])
        mark_articles_as_sent(fresh)

        return {"status": f"Digest sent successfully ({len(fresh)} stories sent and recorded)"}
    except Exception as e:
        err_msg = f"Failed to send digest to Telegram: {str(e)}"
        print(err_msg, file=sys.stderr)
        return {"status": err_msg}


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("fetch_news", fetch_news)
    builder.add_node("filter_new", filter_new)
    builder.add_node("research", research_stories)
    builder.add_node("summarize", summarize)
    builder.add_node("send", send)

    builder.add_edge(START, "fetch_news")
    builder.add_edge("fetch_news", "filter_new")
    builder.add_conditional_edges(
        "filter_new",
        route_after_filter,
        {
            "end": END,
            "research": "research",
        },
    )
    builder.add_edge("research", "summarize")
    builder.add_edge("summarize", "send")
    builder.add_edge("send", END)

    return builder.compile()


async def run_agent() -> Dict[str, Any]:
    """Execute the full agent workflow once."""
    app = build_graph()
    initial_state: AgentState = {
        "news": [],
        "fresh_news": [],
        "research_notes": {},
        "digest": "",
        "status": "Starting",
    }
    result = await app.ainvoke(initial_state)
    return result


if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(f"Status: {result.get('status')}")
