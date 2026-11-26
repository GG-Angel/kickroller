import asyncio

import uvicorn
from prometheus_client import make_asgi_app

from api.core.config import Config
from api.core.state import AppState
from api.server import create_app_with_broker, open_state
from kickroller.logging import configure_logging


async def main() -> None:
    config = Config()
    configure_logging()
    async with open_state(config) as state:
        await serve(config, state)  # share db pool, storage, and broker


async def serve(config: Config, state: AppState) -> None:
    """Run the API and metrics servers until shutdown."""
    api_server = uvicorn.Server(
        uvicorn.Config(
            app=create_app_with_broker(
                state, serve_docs=config.server.serve_docs
            ),
            host=str(config.server.host),
            port=config.server.port,
            log_config=None,  # Handled by loguru
        )
    )
    metrics_server = uvicorn.Server(
        uvicorn.Config(
            app=make_asgi_app(),
            host=str(config.server.host),
            port=config.server.metrics_port,
            log_config=None,  # Handled by loguru
            lifespan="off",  # Has no lifespan; avoids uvicorn's "unsupported" log
            access_log=False,  # Scrapes would flood the logs
        )
    )
    await asyncio.gather(api_server.serve(), metrics_server.serve())


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
