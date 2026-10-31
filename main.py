"""
Main CLI entrypoint for TechNews-MCP.

Usage:
    python main.py run                  # Run LangGraph news digest once
    python main.py schedule --hours 1   # Run recurring scheduler and Telegram bot
    python main.py chat                 # Run 24/7 Telegram chat listener
    python main.py connect              # Auto-connect Telegram bot and detect Chat ID
    python main.py server               # Start FastMCP server on stdio
"""

import argparse
import asyncio
import sys

from src.agent.graph import run_agent
from src.scheduler import run_all, schedule_loop
from src.bot.telegram_bot import poll_telegram_updates
from src.bot.connect_telegram import main as connect_main
from src.mcp_server.server import mcp


def main():
    parser = argparse.ArgumentParser(
        description="TechNews-MCP: Autonomous AI Tech Intelligence Agent & Telegram Companion"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the LangGraph news agent once")

    # Schedule command
    sched_parser = subparsers.add_parser("schedule", help="Run recurring scheduler and Telegram bot")
    sched_parser.add_argument("--interval", "-i", type=int, default=60, help="Interval in minutes (default: 60)")
    sched_parser.add_argument("--hours", type=float, default=None, help="Interval in hours (e.g. --hours 1)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Run interactive Telegram chat listener only")

    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Link Telegram bot and detect Chat ID")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run FastMCP server on stdio")

    args = parser.parse_args()

    # Default to 'run' if no subcommand provided
    command = args.command or "run"

    if command == "run":
        res = asyncio.run(run_agent())
        print(f"Status: {res.get('status', 'Completed')}")
    elif command == "schedule":
        interval = int(args.hours * 60) if args.hours else args.interval
        try:
            asyncio.run(run_all(interval))
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
    elif command == "chat":
        try:
            asyncio.run(poll_telegram_updates())
        except KeyboardInterrupt:
            print("\nTelegram bot stopped.")
    elif command == "connect":
        asyncio.run(connect_main())
    elif command == "server":
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
