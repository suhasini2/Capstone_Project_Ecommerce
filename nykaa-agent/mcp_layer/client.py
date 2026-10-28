"""Separate MCP client process (Task 14).  Run: python -m mcp_layer.client NYK-1003 NYK-1010"""
import asyncio
import os
import sys

from fastmcp import Client

URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")


async def main(ids):
    async with Client(URL) as client:
        print("tools exposed by server:", [t.name for t in await client.list_tools()])
        for rid in ids:
            result = await client.call_tool("check_order_status", {"record_id": rid})
            print(f"\n--- check_order_status({rid!r}) : standardized MCP response ---")
            print(result)
            print("parsed data:", getattr(result, "data", None))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["NYK-1003", "NYK-1010"]))
