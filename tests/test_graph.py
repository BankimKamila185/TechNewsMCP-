"""
Unit tests for LangGraph agent routing and helper utilities.
"""

from graph import (
    AgentState,
    build_graph,
    classify_article,
    extract_mcp_text,
    route_after_filter,
    sanitize_telegram_html,
    summarize,
)
import pytest


def test_extract_mcp_text_various_formats():
    """Verify that extract_mcp_text handles strings, dicts, and list of blocks."""
    # Direct string
    assert extract_mcp_text("plain string") == "plain string"

    # Single dict
    assert extract_mcp_text({"text": "dict text"}) == "dict text"

    # List of strings
    assert extract_mcp_text(["hello", " ", "world"]) == "hello world"

    # List of MCP content block dicts
    blocks = [
        {"type": "text", "text": "Part 1"},
        {"type": "text", "text": " and Part 2"},
    ]
    assert extract_mcp_text(blocks) == "Part 1 and Part 2"

    # Class with .text attribute (mimicking TextContent)
    class DummyBlock:
        def __init__(self, text):
            self.text = text

    assert extract_mcp_text([DummyBlock("Block A")]) == "Block A"


def test_sanitize_telegram_html():
    """Verify that raw & is escaped to &amp; without breaking valid entities."""
    raw = "TechCrunch & The Verge &amp; Ars Technica <a href='https://example.com?a=1&b=2'>link</a>"
    sanitized = sanitize_telegram_html(raw)

    assert "TechCrunch &amp; The Verge" in sanitized
    assert "Verge &amp; Ars" in sanitized
    # Did not double-escape &amp;
    assert "&amp;amp;" not in sanitized
    assert "a=1&amp;b=2" in sanitized


def test_route_after_filter():
    """Test conditional edge routing."""
    empty_state: AgentState = {
        "news": [],
        "fresh_news": [],
        "research_notes": {},
        "digest": "",
        "status": "No new stories",
    }
    assert route_after_filter(empty_state) == "end"

    active_state: AgentState = {
        "news": [{"title": "News 1", "url": "https://example.com/1"}],
        "fresh_news": [{"title": "News 1", "url": "https://example.com/1"}],
        "research_notes": {},
        "digest": "",
        "status": "Found 1 new story",
    }
    assert route_after_filter(active_state) == "research"


def test_graph_structure():
    """Verify that the StateGraph compiles properly with expected nodes and edges."""
    app = build_graph()
    assert app is not None
    assert "fetch_news" in app.nodes
    assert "filter_new" in app.nodes
    assert "research" in app.nodes
    assert "summarize" in app.nodes
    assert "send" in app.nodes


def test_classify_article_categories():
    """Verify classification into user requested categories."""
    # Cyber Crime
    assert classify_article("Zero-day vulnerability exploited in popular firewall") == "cyber_crime"
    assert classify_article("Massive ransomware attack cripples hospital network") == "cyber_crime"

    # New Models & AI
    assert classify_article("Anthropic releases Claude 3.7 Sonnet reasoning model") == "new_models"
    assert classify_article("OpenAI announces new multimodal LLM architecture") == "new_models"

    # New Tech & Hardware
    assert classify_article("Nvidia announces Blackwell Ultra GPU with liquid cooling") == "new_tech"
    assert classify_article("Anduril develops autonomous defense drone hardware") == "new_tech"

    # New Innovation
    assert classify_article("Quantum teleportation breakthrough over optical fiber") == "new_innovation"
    assert classify_article("Biotech scientists engineer plastic-eating enzyme discovery") == "new_innovation"


@pytest.mark.asyncio
async def test_categorized_digest_formatting():
    """Verify that extractive summary organizes stories into categorized sections."""
    sample_state: AgentState = {
        "news": [],
        "fresh_news": [
            {
                "title": "Critical zero-day security flaw patched in OpenSSH",
                "url": "https://example.com/cve-openssh",
                "summary": "Security team addresses remote code execution flaw.",
            },
            {
                "title": "DeepSeek reveals open weights reasoning model",
                "url": "https://example.com/deepseek-r1",
                "summary": "New model matches frontier benchmarks with reinforcement learning.",
            },
            {
                "title": "Qualcomm announces Snapdragon X Elite laptop chip",
                "url": "https://example.com/snapdragon",
                "summary": "New ARM processor challenges Apple silicon in power efficiency.",
            },
        ],
        "research_notes": {},
        "digest": "",
        "status": "",
    }

    res = await summarize(sample_state)
    digest = res["digest"]

    # Verify that category headers appear in the digest
    assert "Cyber Crime" in digest
    assert "New Models" in digest
    assert "New Tech" in digest

    # Verify links and bold formatting
    assert '<a href="https://example.com/cve-openssh">' in digest
    assert '<a href="https://example.com/deepseek-r1">' in digest

