import asyncio

import uvicorn

from src.core.config import CONFIG
from src.server import create_app


async def main() -> None:
    server = uvicorn.Server(uvicorn.Config(app=create_app(CONFIG)))
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
