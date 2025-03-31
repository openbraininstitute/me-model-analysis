"""Main entry point."""

import asyncio
import contextlib

import uvicorn
import uvloop

from app.config import settings
from app.logger import configure_logging


async def main() -> None:
    """Init and run all the async tasks."""
    configure_logging()

    server = uvicorn.Server(
        uvicorn.Config(
            "app.application:app",
            host="0.0.0.0",
            port=settings.UVICORN_PORT,
            proxy_headers=True,
            log_config=None,
        )
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(server.serve(), name="uvicorn")


with contextlib.suppress(KeyboardInterrupt):
    uvloop.run(main())
