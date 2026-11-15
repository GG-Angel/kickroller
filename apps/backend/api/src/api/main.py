import asyncio

import uvicorn
from prometheus_client import make_asgi_app

from api.core.config import Config
from api.server import create_app


async def main() -> None:
    config = Config()
    api = uvicorn.Server(
        uvicorn.Config(
            app=create_app(config),
            host=config.server.host,
            port=config.server.port,
            log_config=None,  # Handled by loguru
        )
    )
    metrics = uvicorn.Server(
        uvicorn.Config(
            app=make_asgi_app(),
            host=config.server.host,
            port=config.server.metrics_port,
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
