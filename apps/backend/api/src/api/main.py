import asyncio

import uvicorn
from prometheus_client import make_asgi_app

from api.core.config import Config
from api.core.logging import configure_logging
from api.core.state import AppState
from api.server import create_app, create_internal_app, open_state


async def main() -> None:
    config = Config()
    configure_logging()
    async with open_state(config) as state:
        await serve(config, state)  # share db pool and storage client


async def serve(config: Config, state: AppState) -> None:
    """Run the public, internal, and metrics servers until shutdown."""
    api = uvicorn.Server(
        uvicorn.Config(
            app=create_app(state),
            host=config.server.host,
            port=config.server.port,
            log_config=None,  # Handled by loguru
        )
    )
    internal = uvicorn.Server(
        uvicorn.Config(
            app=create_internal_app(state),
            host=config.server.host,
            port=config.server.internal_port,
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
    await asyncio.gather(api.serve(), internal.serve(), metrics.serve())


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
