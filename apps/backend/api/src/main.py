import asyncio

import uvicorn

from src.core.config import CONFIG
from src.server import create_app


async def main() -> None:
    server = uvicorn.Server(
        uvicorn.Config(
            app=create_app(CONFIG),
            host=CONFIG.server.host,
            port=CONFIG.server.port,
        )
    )
    await server.serve()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
