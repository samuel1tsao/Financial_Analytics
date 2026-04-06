import asyncio
from main import on_startup

async def main():
    on_startup()

if __name__ == "__main__":
    asyncio.run(main())
