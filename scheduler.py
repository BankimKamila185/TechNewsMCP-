"""
Continuous background runner for tech-news-mcp.
Executes the LangGraph news & research agent every 1 hour (or custom interval).
"""

import argparse
import asyncio
import datetime
import logging
import sys
import time

from graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
        asyncio.run(schedule_loop(interval))
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user.")


if __name__ == "__main__":
    main()

