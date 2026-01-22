"""WhatsApp MCP - Semantic search for WhatsApp conversations.

This module provides tools for parsing WhatsApp chat exports and
exposing them via the Model Context Protocol (MCP) for semantic search.
"""

from whatsapp_mcp.models import Chat, Message
from whatsapp_mcp.parser import parse_whatsapp_file, parse_whatsapp_text, parse_whatsapp_zip

__version__ = "0.1.0"

__all__ = [
    # Models
    "Chat",
    "Message",
    # Parser
    "parse_whatsapp_file",
    "parse_whatsapp_text",
    "parse_whatsapp_zip",
    # Embeddings (lazy import to avoid loading heavy dependencies)
    "EmbeddingService",
    "get_embedding_service",
    "encode_text",
    "encode_texts",
    # Milvus client (lazy import to avoid loading heavy dependencies)
    "MilvusClient",
    "get_milvus_client",
    # Server (lazy import to avoid loading heavy dependencies)
    "mcp",
    "main",
]


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies.

    This allows the package to be imported without requiring
    sentence-transformers or pymilvus to be installed.
    """
    if name in ("EmbeddingService", "get_embedding_service", "encode_text", "encode_texts"):
        from whatsapp_mcp.embeddings import (
            EmbeddingService,
            encode_text,
            encode_texts,
            get_embedding_service,
        )

        return {
            "EmbeddingService": EmbeddingService,
            "get_embedding_service": get_embedding_service,
            "encode_text": encode_text,
            "encode_texts": encode_texts,
        }[name]

    if name in ("MilvusClient", "get_milvus_client"):
        from whatsapp_mcp.milvus_client import MilvusClient, get_milvus_client

        return {
            "MilvusClient": MilvusClient,
            "get_milvus_client": get_milvus_client,
        }[name]

    if name in ("mcp", "main"):
        from whatsapp_mcp.server import main, mcp

        return {
            "mcp": mcp,
            "main": main,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
