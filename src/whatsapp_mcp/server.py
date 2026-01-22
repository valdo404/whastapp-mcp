"""FastMCP server for WhatsApp semantic search.

This module provides an MCP server that exposes semantic search capabilities
for WhatsApp conversations stored in Milvus vector database.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastmcp import FastMCP

from whatsapp_mcp.embeddings import EmbeddingService
from whatsapp_mcp.milvus_client import MilvusClient

# Configure logging to stderr (required for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Global state for services
_milvus_client: MilvusClient | None = None
_embedding_service: EmbeddingService | None = None


@asynccontextmanager
async def lifespan(app: FastMCP) -> Any:
    """Lifecycle manager for the MCP server.

    Handles startup and shutdown of Milvus connection and embedding service.
    """
    global _milvus_client, _embedding_service

    logger.info("Starting WhatsApp MCP Server...")

    # Initialize Milvus client
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    _milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

    try:
        _milvus_client.connect()
        _milvus_client.load_collection()
        logger.info("Connected to Milvus at %s:%d", milvus_host, milvus_port)
    except Exception as e:
        logger.warning("Failed to connect to Milvus: %s", e)
        logger.warning("Server will start but search tools may not work until Milvus is available")

    # Initialize embedding service (lazy loading)
    _embedding_service = EmbeddingService()
    logger.info("Embedding service initialized (model will load on first use)")

    logger.info("WhatsApp MCP Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down WhatsApp MCP Server...")
    if _milvus_client:
        try:
            _milvus_client.release_collection()
            _milvus_client.disconnect()
        except Exception as e:
            logger.warning("Error during Milvus disconnect: %s", e)

    logger.info("WhatsApp MCP Server shut down")


# Create FastMCP server instance
mcp = FastMCP(
    name="WhatsApp MCP Server",
    instructions="""
    This MCP server provides semantic search capabilities for WhatsApp conversations.

    Available tools:
    - semantic_search: Find messages by meaning using natural language queries
    - search_by_date: Search within specific date ranges
    - search_by_sender: Filter messages by participant
    - list_chats: View all available conversations
    - get_chat_stats: Get statistics about the indexed data

    Tips for effective searches:
    - Use descriptive queries like "discussions about vacation plans"
    - Combine date filters with semantic search for precise results
    - Use list_chats first to see available conversations
    """,
    lifespan=lifespan,
)


def _format_timestamp(timestamp: int | None) -> str | None:
    """Format Unix timestamp to ISO format string."""
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp).isoformat()
    except (ValueError, OSError):
        return None


def _format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format search results for AI assistant consumption."""
    formatted = []
    for result in results:
        formatted.append({
            "content": result.get("content"),
            "sender": result.get("sender"),
            "timestamp": _format_timestamp(result.get("timestamp")),
            "chat_id": result.get("chat_id"),
            "chat_name": result.get("chat_name"),
            "relevance_score": result.get("score"),
        })
    return formatted


@mcp.tool
def semantic_search(
    query: str,
    limit: int = 10,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Search WhatsApp messages using semantic similarity.

    Finds messages that are semantically similar to the query,
    even if they do not contain the exact words.

    Args:
        query: Natural language search query.
               Examples: "vacation plans", "birthday party", "work meeting"
        limit: Maximum number of results to return (1-50, default: 10)
        chat_id: Optional chat ID to limit search to a specific conversation.
                 Use list_chats() to see available chat IDs.

    Returns:
        Dictionary with:
        - success: Whether the search was successful
        - results: List of matching messages with content, sender, timestamp,
                   chat_id, chat_name, and relevance_score (0-1, higher is better)
        - error: Error message if search failed
    """
    if not _milvus_client or not _embedding_service:
        return {
            "success": False,
            "error": "Server not properly initialized. Milvus connection may be unavailable.",
            "results": [],
        }

    if not query or not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty",
            "results": [],
        }

    # Enforce limits
    limit = max(1, min(limit, 50))

    try:
        # Generate query embedding
        query_embedding = _embedding_service.encode(query)

        # Search Milvus
        results = _milvus_client.search(
            query_embedding=query_embedding,
            limit=limit,
            chat_id=chat_id,
        )

        return {
            "success": True,
            "query": query,
            "results": _format_search_results(results),
            "total_results": len(results),
        }
    except Exception as e:
        logger.exception("Error during semantic search")
        return {
            "success": False,
            "error": f"Search failed: {e!s}",
            "results": [],
        }


@mcp.tool
def search_by_date(
    start_date: str,
    end_date: str,
    query: str | None = None,
    chat_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search messages within a specific date range.

    Can be combined with semantic search for more precise results.

    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        query: Optional semantic search query to filter and rank results
        chat_id: Optional chat ID to limit search
        limit: Maximum number of results (default: 20)

    Returns:
        Dictionary with:
        - success: Whether the search was successful
        - results: List of messages within the date range, optionally
                   ranked by semantic relevance if query is provided
        - error: Error message if search failed
    """
    if not _milvus_client:
        return {
            "success": False,
            "error": "Server not properly initialized. Milvus connection may be unavailable.",
            "results": [],
        }

    # Parse dates
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format. Use YYYY-MM-DD. Error: {e!s}",
            "results": [],
        }

    # Enforce limits
    limit = max(1, min(limit, 100))

    try:
        # Generate query embedding if query provided
        query_embedding = None
        if query and _embedding_service:
            query_embedding = _embedding_service.encode(query)

        # Search Milvus
        results = _milvus_client.search_by_date(
            start_date=start_dt,
            end_date=end_dt,
            query_embedding=query_embedding,
            chat_id=chat_id,
            limit=limit,
        )

        return {
            "success": True,
            "date_range": {"start": start_date, "end": end_date},
            "query": query,
            "results": _format_search_results(results),
            "total_results": len(results),
        }
    except Exception as e:
        logger.exception("Error during date search")
        return {
            "success": False,
            "error": f"Search failed: {e!s}",
            "results": [],
        }


@mcp.tool
def search_by_sender(
    sender: str,
    query: str | None = None,
    chat_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search messages from a specific sender.

    Args:
        sender: Name of the sender to filter by (exact match)
        query: Optional semantic search query to filter and rank results
        chat_id: Optional chat ID to limit search
        limit: Maximum number of results (default: 20)

    Returns:
        Dictionary with:
        - success: Whether the search was successful
        - results: List of messages from the specified sender,
                   optionally ranked by semantic relevance
        - error: Error message if search failed
    """
    if not _milvus_client:
        return {
            "success": False,
            "error": "Server not properly initialized. Milvus connection may be unavailable.",
            "results": [],
        }

    if not sender or not sender.strip():
        return {
            "success": False,
            "error": "Sender name cannot be empty",
            "results": [],
        }

    # Enforce limits
    limit = max(1, min(limit, 100))

    try:
        # Generate query embedding if query provided
        query_embedding = None
        if query and _embedding_service:
            query_embedding = _embedding_service.encode(query)

        # Search Milvus
        results = _milvus_client.search_by_sender(
            sender=sender,
            query_embedding=query_embedding,
            chat_id=chat_id,
            limit=limit,
        )

        return {
            "success": True,
            "sender": sender,
            "query": query,
            "results": _format_search_results(results),
            "total_results": len(results),
        }
    except Exception as e:
        logger.exception("Error during sender search")
        return {
            "success": False,
            "error": f"Search failed: {e!s}",
            "results": [],
        }


@mcp.tool
def list_chats() -> dict[str, Any]:
    """List all available WhatsApp conversations.

    Returns:
        Dictionary with:
        - success: Whether the operation was successful
        - chats: List of chats with:
            - chat_id: Unique identifier for the chat
            - chat_name: Display name of the conversation
            - message_count: Total number of messages
            - participants: List of participant names
            - date_range: First and last message dates
        - error: Error message if operation failed
    """
    if not _milvus_client:
        return {
            "success": False,
            "error": "Server not properly initialized. Milvus connection may be unavailable.",
            "chats": [],
        }

    try:
        chats = _milvus_client.list_chats()

        return {
            "success": True,
            "chats": chats,
            "total_chats": len(chats),
        }
    except Exception as e:
        logger.exception("Error listing chats")
        return {
            "success": False,
            "error": f"Failed to list chats: {e!s}",
            "chats": [],
        }


@mcp.tool
def get_chat_stats() -> dict[str, Any]:
    """Get statistics about the indexed WhatsApp data.

    Returns:
        Dictionary with:
        - success: Whether the operation was successful
        - stats: Statistics including:
            - collection_name: Name of the Milvus collection
            - exists: Whether the collection exists
            - num_entities: Total number of indexed messages
        - error: Error message if operation failed
    """
    if not _milvus_client:
        return {
            "success": False,
            "error": "Server not properly initialized. Milvus connection may be unavailable.",
            "stats": {},
        }

    try:
        stats = _milvus_client.get_collection_stats()

        return {
            "success": True,
            "stats": stats,
        }
    except Exception as e:
        logger.exception("Error getting chat stats")
        return {
            "success": False,
            "error": f"Failed to get stats: {e!s}",
            "stats": {},
        }


def main() -> None:
    """Entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="WhatsApp MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio for Claude Desktop)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")

    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
