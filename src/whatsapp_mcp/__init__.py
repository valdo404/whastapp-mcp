"""WhatsApp MCP - Semantic search for WhatsApp conversations.

This module provides tools for parsing WhatsApp chat exports and
exposing them via the Model Context Protocol (MCP) for semantic search.
"""

from whatsapp_mcp.models import Chat, Message
from whatsapp_mcp.parser import parse_whatsapp_text, parse_whatsapp_zip

__version__ = "0.1.0"

__all__ = [
    "Chat",
    "Message",
    "parse_whatsapp_text",
    "parse_whatsapp_zip",
]
