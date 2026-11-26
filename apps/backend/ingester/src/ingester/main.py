import asyncio

from faststream import FastStream

from ingester.core.broker import build_broker
from ingester.core.config import Config
from ingester.core.logging import configure_logging
from ingester.core.storage import create_store


async def main() -> None:
    config = Config()
    configure_logging()
    # Fail startup if RustFS is down.
    store = await create_store(config.storage)
    app = FastStream(build_broker(config.broker))
    app.context.set_global("store", store)  # Handlers take Context("store").
    await app.run()  # Connects to RabbitMQ and runs until SIGINT/SIGTERM.


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
