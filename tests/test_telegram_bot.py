"""
Unit tests for conversational Telegram bot handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from telegram_bot import (
    generate_conversational_reply,
    get_db_stats,
    handle_user_command,
    process_telegram_update,
    reply_telegram_message,
)


@pytest.mark.asyncio
async def test_generate_conversational_reply_ai_pace():
    """Verify AI pace inquiries return insightful responses."""
    resp = await generate_conversational_reply("ai is getting to far", user_name="Bankim")
    assert "breathtaking pace" in resp
    assert "Test-Time Reasoning" in resp or "Autonomous" in resp


@pytest.mark.asyncio
async def test_generate_conversational_reply_cybersecurity():
    """Verify cybersecurity topics trigger specialized security context or live search."""
    resp = await generate_conversational_reply("Tell me about latest cyber security attacks", user_name="Bankim")
    assert "Cyber" in resp or "cyber" in resp or "attack" in resp or "zero-trust" in resp


@pytest.mark.asyncio
async def test_generate_conversational_reply_hardware():
    """Verify hardware topics trigger semiconductor and compute responses."""
    resp = await generate_conversational_reply("What is the latest with nvidia and intel chips?", user_name="Bankim")
    assert "Nvidia" in resp or "Blackwell" in resp or "Hardware" in resp or "Silicon" in resp or "compute" in resp


@pytest.mark.asyncio
async def test_handle_user_command_start():
    """Verify /start returns greeting and command list."""
    res = await handle_user_command("/start", "", 12345, "Bankim")
    assert "Welcome to TechNews Bot" in res
    assert "/news" in res
    assert "/search" in res


@pytest.mark.asyncio
async def test_handle_user_command_status():
    """Verify /status returns system statistics."""
    res = await handle_user_command("/status", "", 12345, "Bankim")
    assert "TechNews System Status" in res
    assert "Operational" in res


@pytest.mark.asyncio
async def test_handle_user_command_categories():
    """Verify /categories lists 5 topic sections."""
    res = await handle_user_command("/categories", "", 12345, "Bankim")
    assert "New Models" in res
    assert "Cyber Crime" in res
    assert "New Tech" in res


@pytest.mark.asyncio
async def test_handle_user_command_search_empty():
    """Verify /search without arguments prompts for a topic."""
    res = await handle_user_command("/search", "", 12345, "Bankim")
    assert "Please specify what you would like to search for" in res


@pytest.mark.asyncio
async def test_handle_user_command_search_mocked():
    """Verify /search formats returned search results."""
    mock_results = '{"results": [{"title": "DeepSeek R1", "snippet": "New open-weights reasoning model", "url": "https://deepseek.com"}]}'
    with patch("telegram_bot.web_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results
        res = await handle_user_command("/search", "deepseek", 12345, "Bankim")
        assert "Search results for 'deepseek'" in res
        assert "DeepSeek R1" in res


@pytest.mark.asyncio
async def test_process_telegram_update_conversational_message():
    """Verify incoming conversational message is processed and replied to."""
    fake_update = {
        "update_id": 9999,
        "message": {
            "message_id": 42,
            "chat": {"id": 12345},
            "from": {"first_name": "Bankim", "username": "Bankim5"},
            "text": "ai is getting to far",
        },
    }
    with patch("telegram_bot.reply_telegram_message", new_callable=AsyncMock) as mock_reply, \
         patch("telegram_bot.send_typing_action", new_callable=AsyncMock):
        mock_reply.return_value = True
        await process_telegram_update(fake_update, "dummy-token")
        assert mock_reply.called
        args, kwargs = mock_reply.call_args
        # kwargs or args: token, chat_id, text, reply_to_message_id
        assert args[1] == 12345
        assert "breathtaking pace" in args[2]
