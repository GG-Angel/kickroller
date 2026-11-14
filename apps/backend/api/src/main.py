import asyncio

from src.core.logging import configure_logging


async def main() -> None:
    configure_logging()


if __name__ == "__main__":
    asyncio.run(main())
