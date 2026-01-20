"""Milvus client for vector database operations.

This module provides a client for interacting with Milvus vector database,
including collection management, data insertion, and semantic search.
"""

import logging
import os
from datetime import datetime
from typing import Any

from whatsapp_mcp.embeddings import DEFAULT_DIMENSION
from whatsapp_mcp.models import Message

logger = logging.getLogger(__name__)

# Default Milvus connection settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 19530
DEFAULT_COLLECTION = "whatsapp_messages"

# Schema field sizes
MAX_CONTENT_LENGTH = 65535
MAX_VARCHAR_LENGTH = 255


class MilvusClient:
    """Client for interacting with Milvus vector database.

    This client provides methods for:
    - Connection management
    - Collection creation with proper schema
    - Index creation (HNSW with COSINE similarity)
    - Data insertion with embeddings
    - Semantic search with optional filtering

    Attributes:
        host: Milvus server hostname.
        port: Milvus server port.
        collection_name: Name of the collection to use.

    Example:
        >>> client = MilvusClient()
        >>> client.connect()
        >>> client.ensure_collection()
        >>> client.insert_messages(messages, embeddings)
        >>> results = client.search("vacation plans", limit=10)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the Milvus client.

        Args:
            host: Milvus server hostname. Defaults to MILVUS_HOST env var or 'localhost'.
            port: Milvus server port. Defaults to MILVUS_PORT env var or 19530.
            collection_name: Name of the collection. Defaults to 'whatsapp_messages'.
        """
        self.host = host or os.getenv("MILVUS_HOST", DEFAULT_HOST)
        self.port = port or int(os.getenv("MILVUS_PORT", str(DEFAULT_PORT)))
        self.collection_name = collection_name or DEFAULT_COLLECTION
        self._collection: Any = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the Milvus server.

        Raises:
            ImportError: If pymilvus is not installed.
            ConnectionError: If connection to Milvus fails.
        """
        if self._connected:
            return

        try:
            from pymilvus import connections
        except ImportError as e:
            raise ImportError(
                "pymilvus is required for Milvus operations. "
                "Install it with: pip install pymilvus"
            ) from e

        logger.info("Connecting to Milvus at %s:%d", self.host, self.port)

        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            self._connected = True
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the Milvus server."""
        if not self._connected:
            return

        from pymilvus import connections

        connections.disconnect("default")
        self._connected = False
        self._collection = None
        logger.info("Disconnected from Milvus")

    def ensure_collection(self, dimension: int = DEFAULT_DIMENSION) -> None:
        """Ensure the collection exists with the correct schema.

        Creates the collection if it doesn't exist, including all necessary
        indexes for efficient search.

        Args:
            dimension: The dimension of the embedding vectors.
        """
        self.connect()

        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        # Check if collection already exists
        if utility.has_collection(self.collection_name):
            logger.info("Collection '%s' already exists", self.collection_name)
            self._collection = Collection(self.collection_name)
            return

        logger.info("Creating collection '%s'", self.collection_name)

        # Define schema fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Auto-generated message ID",
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=MAX_CONTENT_LENGTH,
                description="Message text content",
            ),
            FieldSchema(
                name="sender",
                dtype=DataType.VARCHAR,
                max_length=MAX_VARCHAR_LENGTH,
                description="Name of the message sender",
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.INT64,
                description="Message timestamp (Unix epoch seconds)",
            ),
            FieldSchema(
                name="chat_id",
                dtype=DataType.VARCHAR,
                max_length=MAX_VARCHAR_LENGTH,
                description="Unique identifier for the conversation",
            ),
            FieldSchema(
                name="chat_name",
                dtype=DataType.VARCHAR,
                max_length=MAX_VARCHAR_LENGTH,
                description="Human-readable chat name",
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
                description="Semantic embedding vector",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="WhatsApp chat messages with semantic embeddings",
            enable_dynamic_field=False,
        )

        # Create collection
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Bounded",
        )

        # Create indexes
        self._create_indexes()

        logger.info("Collection '%s' created successfully", self.collection_name)

    def _create_indexes(self) -> None:
        """Create indexes on the collection for efficient search."""
        if self._collection is None:
            return

        logger.info("Creating indexes on collection '%s'", self.collection_name)

        # Vector index - HNSW for fast approximate nearest neighbor search
        vector_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 16,  # Max connections per node
                "efConstruction": 256,  # Build-time search width
            },
        }

        self._collection.create_index(
            field_name="embedding",
            index_params=vector_index_params,
            index_name="embedding_hnsw_idx",
        )

        # Scalar indexes for filtering
        self._collection.create_index(
            field_name="sender",
            index_params={"index_type": "INVERTED"},
            index_name="sender_idx",
        )

        self._collection.create_index(
            field_name="timestamp",
            index_params={"index_type": "STL_SORT"},
            index_name="timestamp_idx",
        )

        self._collection.create_index(
            field_name="chat_id",
            index_params={"index_type": "INVERTED"},
            index_name="chat_id_idx",
        )

        logger.info("Indexes created successfully")

    def load_collection(self) -> None:
        """Load the collection into memory for searching."""
        if self._collection is None:
            self.ensure_collection()

        if self._collection is not None:
            self._collection.load()
            logger.info("Collection '%s' loaded into memory", self.collection_name)

    def release_collection(self) -> None:
        """Release the collection from memory."""
        if self._collection is not None:
            self._collection.release()
            logger.info("Collection '%s' released from memory", self.collection_name)

    def insert(
        self,
        messages: list[Message],
        embeddings: list[list[float]],
    ) -> int:
        """Insert messages with their embeddings into the collection.

        Args:
            messages: List of Message objects to insert.
            embeddings: List of embedding vectors corresponding to the messages.

        Returns:
            Number of entities inserted.

        Raises:
            ValueError: If the number of messages and embeddings don't match.
        """
        if len(messages) != len(embeddings):
            raise ValueError(
                f"Number of messages ({len(messages)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not messages:
            return 0

        self.ensure_collection()

        # Prepare data for insertion as list of lists (column-based format)
        # Order must match schema: content, sender, timestamp, chat_id, chat_name, embedding
        data = [
            [self._truncate_content(msg.content) for msg in messages],
            [msg.sender[:MAX_VARCHAR_LENGTH] for msg in messages],
            [int(msg.timestamp.timestamp()) for msg in messages],
            [msg.chat_id[:MAX_VARCHAR_LENGTH] for msg in messages],
            [msg.chat_name[:MAX_VARCHAR_LENGTH] for msg in messages],
            embeddings,
        ]

        if self._collection is not None:
            result = self._collection.insert(data)
            logger.debug("Inserted %d messages", len(messages))
            return int(result.insert_count)

        return 0

    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit within the maximum length.

        Args:
            content: The content to truncate.

        Returns:
            Truncated content.
        """
        if len(content) <= MAX_CONTENT_LENGTH:
            return content
        return content[: MAX_CONTENT_LENGTH - 3] + "..."

    def flush(self) -> None:
        """Flush pending data to storage."""
        if self._collection is not None:
            self._collection.flush()
            logger.info("Collection '%s' flushed", self.collection_name)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        chat_id: str | None = None,
        sender: str | None = None,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar messages.

        Args:
            query_embedding: The embedding vector of the search query.
            limit: Maximum number of results to return.
            chat_id: Optional filter by chat ID.
            sender: Optional filter by sender name.
            start_timestamp: Optional filter for messages after this timestamp.
            end_timestamp: Optional filter for messages before this timestamp.

        Returns:
            List of matching messages with metadata and similarity scores.
        """
        self.load_collection()

        # Build filter expression
        filters = []
        if chat_id:
            filters.append(f'chat_id == "{chat_id}"')
        if sender:
            filters.append(f'sender == "{sender}"')
        if start_timestamp:
            filters.append(f"timestamp >= {start_timestamp}")
        if end_timestamp:
            filters.append(f"timestamp <= {end_timestamp}")

        expr = " and ".join(filters) if filters else None

        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64},  # Search-time accuracy
        }

        if self._collection is None:
            return []

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["content", "sender", "timestamp", "chat_id", "chat_name"],
        )

        # Format results
        return [
            {
                "content": hit.entity.get("content"),
                "sender": hit.entity.get("sender"),
                "timestamp": hit.entity.get("timestamp"),
                "chat_id": hit.entity.get("chat_id"),
                "chat_name": hit.entity.get("chat_name"),
                "score": round(hit.score, 4),
            }
            for hit in results[0]
        ]

    def search_by_date(
        self,
        start_date: datetime,
        end_date: datetime,
        query_embedding: list[float] | None = None,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search messages within a date range.

        Args:
            start_date: Start of the date range.
            end_date: End of the date range.
            query_embedding: Optional embedding for semantic ranking.
            chat_id: Optional filter by chat ID.
            limit: Maximum number of results.

        Returns:
            List of messages within the date range.
        """
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        if query_embedding:
            return self.search(
                query_embedding=query_embedding,
                limit=limit,
                chat_id=chat_id,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )

        # Query without semantic search
        self.load_collection()

        filters = [f"timestamp >= {start_ts}", f"timestamp <= {end_ts}"]
        if chat_id:
            filters.append(f'chat_id == "{chat_id}"')

        expr = " and ".join(filters)

        if self._collection is None:
            return []

        results = self._collection.query(
            expr=expr,
            output_fields=["content", "sender", "timestamp", "chat_id", "chat_name"],
            limit=limit,
        )

        return [
            {
                "content": r.get("content"),
                "sender": r.get("sender"),
                "timestamp": r.get("timestamp"),
                "chat_id": r.get("chat_id"),
                "chat_name": r.get("chat_name"),
                "score": None,
            }
            for r in results
        ]

    def search_by_sender(
        self,
        sender: str,
        query_embedding: list[float] | None = None,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search messages from a specific sender.

        Args:
            sender: Name of the sender to filter by.
            query_embedding: Optional embedding for semantic ranking.
            chat_id: Optional filter by chat ID.
            limit: Maximum number of results.

        Returns:
            List of messages from the specified sender.
        """
        if query_embedding:
            return self.search(
                query_embedding=query_embedding,
                limit=limit,
                chat_id=chat_id,
                sender=sender,
            )

        # Query without semantic search
        self.load_collection()

        filters = [f'sender == "{sender}"']
        if chat_id:
            filters.append(f'chat_id == "{chat_id}"')

        expr = " and ".join(filters)

        if self._collection is None:
            return []

        results = self._collection.query(
            expr=expr,
            output_fields=["content", "sender", "timestamp", "chat_id", "chat_name"],
            limit=limit,
        )

        return [
            {
                "content": r.get("content"),
                "sender": r.get("sender"),
                "timestamp": r.get("timestamp"),
                "chat_id": r.get("chat_id"),
                "chat_name": r.get("chat_name"),
                "score": None,
            }
            for r in results
        ]

    def list_chats(self) -> list[dict[str, Any]]:
        """List all available chats with statistics.

        Returns:
            List of chat information including message counts and date ranges.
        """
        self.load_collection()

        if self._collection is None:
            return []

        # Use query iterator to handle large collections
        # First, get a sample to find unique chat_ids
        sample_results = self._collection.query(
            expr="",
            output_fields=["chat_id", "chat_name"],
            limit=16384,
        )

        # Get unique chat_ids from sample
        chat_ids = set()
        chat_names: dict[str, str] = {}
        for r in sample_results:
            chat_id = r.get("chat_id")
            chat_ids.add(chat_id)
            if chat_id not in chat_names:
                chat_names[chat_id] = r.get("chat_name", "")

        # For each chat_id, get statistics
        chats: list[dict[str, Any]] = []
        for chat_id in chat_ids:
            # Get messages for this chat
            chat_results = self._collection.query(
                expr=f'chat_id == "{chat_id}"',
                output_fields=["sender", "timestamp"],
                limit=16384,
            )

            if not chat_results:
                continue

            participants: set[str] = set()
            timestamps: list[int] = []

            for r in chat_results:
                participants.add(r.get("sender", ""))
                timestamps.append(r.get("timestamp", 0))

            if timestamps:
                chats.append({
                    "chat_id": chat_id,
                    "chat_name": chat_names.get(chat_id, ""),
                    "message_count": len(chat_results),
                    "participants": list(participants),
                    "date_range": {
                        "first": datetime.fromtimestamp(min(timestamps)).isoformat(),
                        "last": datetime.fromtimestamp(max(timestamps)).isoformat(),
                    },
                })

        return chats

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        self.connect()

        from pymilvus import utility

        if not utility.has_collection(self.collection_name):
            return {
                "exists": False,
                "collection_name": self.collection_name,
            }

        self.ensure_collection()

        if self._collection is None:
            return {
                "exists": False,
                "collection_name": self.collection_name,
            }

        return {
            "exists": True,
            "collection_name": self.collection_name,
            "num_entities": self._collection.num_entities,
            "schema": str(self._collection.schema),
        }

    def drop_collection(self) -> None:
        """Drop the collection (delete all data).

        Warning: This operation is irreversible!
        """
        self.connect()

        from pymilvus import utility

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self._collection = None
            logger.info("Collection '%s' dropped", self.collection_name)


# Global singleton instance
_default_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """Get the default Milvus client instance.

    Returns:
        The default MilvusClient instance.
    """
    global _default_client
    if _default_client is None:
        _default_client = MilvusClient()
    return _default_client
