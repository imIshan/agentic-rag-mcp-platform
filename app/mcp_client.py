import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_list_sources_tool() -> str:
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "app.mcp_server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_sources", {})

            if result.isError:
                return "Error calling MCP tool."

            if result.content and len(result.content) > 0:
                return result.content[0].text

            return "No indexed sources are currently available."

def get_list_sources_from_mcp() -> str:
    return asyncio.run(call_list_sources_tool())