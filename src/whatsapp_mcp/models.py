"""Pydantic models for WhatsApp chat data.

This module defines the data models used to represent WhatsApp messages
and chats with proper validation using Pydantic.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single WhatsApp message.

    Attributes:
        content: The text content of the message.
        sender: The name of the message sender.
        timestamp: When the message was sent.
        chat_id: Unique identifier for the chat/conversation.
        chat_name: Human-readable name of the chat.
        is_media: Whether this message contains media (image, video, etc.).
        is_system: Whether this is a system message (security code changed, etc.).
        is_deleted: Whether this message was deleted by the user.
        is_edited: Whether this message was edited.
    """

    content: str = Field(..., description="The text content of the message")
    sender: str = Field(..., description="The name of the message sender")
    timestamp: datetime = Field(..., description="When the message was sent")
    chat_id: str = Field(..., description="Unique identifier for the chat/conversation")
    chat_name: str = Field(..., description="Human-readable name of the chat")
    is_media: bool = Field(default=False, description="Whether this message contains media")
    is_system: bool = Field(default=False, description="Whether this is a system message")
    is_deleted: bool = Field(default=False, description="Whether this message was deleted")
    is_edited: bool = Field(default=False, description="Whether this message was edited")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "content": "Bonjour, comment ça va?",
                "sender": "Jean Dupont",
                "timestamp": "2025-03-18T07:37:36",
                "chat_id": "astride_zircon",
                "chat_name": "Astride Zircon",
                "is_media": False,
                "is_system": False,
                "is_deleted": False,
                "is_edited": False,
            }
        }


class Chat(BaseModel):
    """Represents a WhatsApp chat/conversation.

    Attributes:
        chat_id: Unique identifier for the chat.
        chat_name: Human-readable name of the chat (contact or group name).
        messages: List of messages in the chat, ordered by timestamp.
        participants: Set of unique sender names in the chat.
        message_count: Total number of messages in the chat.
        first_message_at: Timestamp of the first message.
        last_message_at: Timestamp of the last message.
    """

    chat_id: str = Field(..., description="Unique identifier for the chat")
    chat_name: str = Field(..., description="Human-readable name of the chat")
    messages: list[Message] = Field(
        default_factory=list, description="List of messages in the chat"
    )

    @property
    def participants(self) -> set[str]:
        """Get unique sender names in the chat."""
        return {msg.sender for msg in self.messages}

    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    @property
    def first_message_at(self) -> datetime | None:
        """Get timestamp of the first message."""
        if not self.messages:
            return None
        return min(msg.timestamp for msg in self.messages)

    @property
    def last_message_at(self) -> datetime | None:
        """Get timestamp of the last message."""
        if not self.messages:
            return None
        return max(msg.timestamp for msg in self.messages)

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "chat_id": "astride_zircon",
                "chat_name": "Astride Zircon",
                "messages": [
                    {
                        "content": "Bonjour",
                        "sender": "Astride Zircon",
                        "timestamp": "2025-03-18T07:37:36",
                        "chat_id": "astride_zircon",
                        "chat_name": "Astride Zircon",
                        "is_media": False,
                        "is_system": False,
                        "is_deleted": False,
                        "is_edited": False,
                    }
                ],
            }
        }
