import asyncio

import uvicorn
from prometheus_client import make_asgi_app

from src.core.config import CONFIG
from src.server import create_app


async def main() -> None:
    api = uvicorn.Server(
        uvicorn.Config(
            app=create_app(CONFIG),
            host=CONFIG.server.host,
            port=CONFIG.server.port,
            log_config=None,  # Handled by loguru
        )
    )
    metrics = uvicorn.Server(
        uvicorn.Config(
            app=make_asgi_app(),
            host=CONFIG.server.host,
            port=CONFIG.server.metrics_port,
            log_config=None,  # Handled by loguru
            lifespan="off",  # Has no lifespan; avoids uvicorn's "unsupported" log
            access_log=False,  # Scrapes would flood the logs
        )
    )
    await asyncio.gather(api.serve(), metrics.serve())


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
