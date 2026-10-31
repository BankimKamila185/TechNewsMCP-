"""
Continuous background runner for tech-news-mcp.
Executes the LangGraph news & research agent every 1 hour (or custom interval).
"""

import argparse
import asyncio
import datetime
import logging
import os
import sys
import time

try:
    from src.agent.graph import run_agent
    from src.bot.telegram_bot import poll_telegram_updates
except ImportError:
    from graph import run_agent
    from telegram_bot import poll_telegram_updates

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOG_DIR = os.path.join(_ROOT_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")


async def schedule_loop(interval_minutes: int = 60):
    """Run agent in an infinite loop every interval_minutes."""
    if interval_minutes >= 60 and interval_minutes % 60 == 0:
        logger.info("🚀 Tech News MCP Scheduler running every %d hour(s).", interval_minutes // 60)
    else:
        logger.info("🚀 Tech News MCP Scheduler running every %d minutes.", interval_minutes)
    interval_seconds = interval_minutes * 60
    iteration = 1

    while True:
        logger.info("▶️ [Cycle #%d] Fetching, researching, and checking new tech news...", iteration)
        t_start = time.time()
        try:
            result = await run_agent()
            status = result.get("status", "Done")
            logger.info("✅ [Cycle #%d] Status: %s", iteration, status)
        except Exception as e:
            logger.error("❌ [Cycle #%d] Error during execution: %s", iteration, e)

        elapsed = time.time() - t_start
        sleep_time = max(0, interval_seconds - elapsed)
        next_run = datetime.datetime.now() + datetime.timedelta(seconds=sleep_time)
        logger.info(
            "⏳ Next hourly check scheduled at %s (in %d minutes).\n",
            next_run.strftime("%H:%M:%S"),
            int(sleep_time // 60),
        )

        try:
            await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            break

        iteration += 1


async def run_all(interval_minutes: int):
    """Run hourly news scheduler and real-time interactive Telegram bot concurrently."""
    await asyncio.gather(
        schedule_loop(interval_minutes),
        poll_telegram_updates(),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Tech News Digest periodically.")
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Interval between checks in minutes (default: 60)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Interval between checks in hours (e.g. --hours 1)",
    )
    args = parser.parse_args()

    interval = int(args.hours * 60) if args.hours else args.interval

    try:
        asyncio.run(run_all(interval))
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user.")


if __name__ == "__main__":
    main()

