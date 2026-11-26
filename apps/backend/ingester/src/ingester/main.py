import asyncio

from ingester.core.config import Config
from ingester.worker import create_app, open_state
from kickroller.logging import configure_logging


async def main() -> None:
    config = Config()
    configure_logging()
    async with open_state(config) as state:
        await create_app(state).run()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
