from app.retrieval import list_available_sources
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-tools")

@mcp.tool()
def list_sources() -> str:
    """Return the list of indexed source documents in the knowledge base."""
    sources = list_available_sources()

    if not sources:
        return "No indexed sources are currently available."

    return "\n".join(sources)

@mcp.tool()
def count_sources() -> int:
    """Return the number of unique indexed source documents."""
    return len(list_available_sources())


if __name__ == "__main__":
    mcp.run()