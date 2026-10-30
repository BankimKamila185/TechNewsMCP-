"""
Web server entrypoint for Render deployment.
Binds to $PORT, provides health checks, manual trigger endpoints,
and runs the hourly Tech News MCP scheduler in the background.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from graph import run_agent
from scheduler import schedule_loop

PORT = int(os.getenv("PORT", "10000"))
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "60"))
ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "true").lower() in ("true", "1", "yes")


async def health_check(request):
    """Health check endpoint for Render monitoring."""
    return JSONResponse({
        "status": "healthy",
        "service": "tech-news-mcp",
        "interval_minutes": CHECK_INTERVAL_MINUTES,
    })


async def trigger_digest(request):
    """On-demand trigger endpoint to run the agent immediately via HTTP."""
    try:
        result = await run_agent()
        return JSONResponse({"status": "success", "result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Manage background scheduler lifecycle across server startup and shutdown."""
    task = None
    if ENABLE_SCHEDULER:
        task = asyncio.create_task(schedule_loop(CHECK_INTERVAL_MINUTES))
    yield
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


routes = [
    Route("/", endpoint=health_check, methods=["GET"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/trigger", endpoint=trigger_digest, methods=["GET", "POST"]),
]

app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
