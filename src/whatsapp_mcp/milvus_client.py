"""Milvus client for vector database operations.

This module provides a client for interacting with Milvus vector database,
including collection management, data insertion, and semantic search.
"""

import hashlib
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
HASH_LENGTH = 64  # SHA-256 hex digest length


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

    @staticmethod
    def compute_message_hash(
        timestamp: int,
        sender: str,
        chat_id: str,
        content: str,
    ) -> str:
        """Compute a unique hash for a message based on its natural key.

        The natural key consists of timestamp, sender, chat_id, and content.
        This hash is used as the primary key for deduplication.

        Args:
            timestamp: Message timestamp (Unix epoch seconds).
            sender: Name of the message sender.
            chat_id: Unique identifier for the conversation.
            content: Message text content.

        Returns:
            A 64-character hex string (SHA-256 hash).
        """
        key = f"{timestamp}|{sender}|{chat_id}|{content}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

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
        # Using message_hash as primary key for deduplication (no auto_id)
        fields = [
            FieldSchema(
                name="message_hash",
                dtype=DataType.VARCHAR,
                max_length=HASH_LENGTH,
                is_primary=True,
                auto_id=False,
                description="SHA-256 hash of (timestamp|sender|chat_id|content) for deduplication",
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
        """Insert or update messages with their embeddings into the collection.

        Uses upsert to handle deduplication based on message_hash (natural key).
        If a message with the same hash already exists, it will be updated.

        Args:
            messages: List of Message objects to insert.
            embeddings: List of embedding vectors corresponding to the messages.

        Returns:
            Number of entities upserted.

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

        # Compute message hashes for deduplication
        timestamps = [int(msg.timestamp.timestamp()) for msg in messages]
        senders = [msg.sender[:MAX_VARCHAR_LENGTH] for msg in messages]
        chat_ids = [msg.chat_id[:MAX_VARCHAR_LENGTH] for msg in messages]
        contents = [self._truncate_content(msg.content) for msg in messages]

        message_hashes = [
            self.compute_message_hash(ts, sender, chat_id, content)
            for ts, sender, chat_id, content in zip(timestamps, senders, chat_ids, contents)
        ]

        # Prepare data for upsert as list of lists (column-based format)
        # Order must match schema: message_hash, content, sender, timestamp, chat_id, chat_name, embedding
        data = [
            message_hashes,
            contents,
            senders,
            timestamps,
            chat_ids,
            [msg.chat_name[:MAX_VARCHAR_LENGTH] for msg in messages],
            embeddings,
        ]

        if self._collection is not None:
            # Use upsert for deduplication - if hash exists, update; otherwise insert
            result = self._collection.upsert(data)
            logger.debug("Upserted %d messages", len(messages))
            return int(result.upsert_count)

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
        # ef must be >= limit for HNSW search to work
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(64, limit)},  # Search-time accuracy, must be >= limit
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
            List of messages within the date range, sorted chronologically.
        """
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        if query_embedding:
            # For semantic search, we need to get more results, sort, then limit
            # Get 10x the limit to ensure we have enough for proper ordering
            raw_results = self.search(
                query_embedding=query_embedding,
                limit=min(limit * 10, 5000),  # Cap at 5000 to avoid performance issues
                chat_id=chat_id,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )
            # Sort by timestamp and apply limit
            sorted_results = sorted(raw_results, key=lambda r: r.get("timestamp", 0))
            return sorted_results[:limit]

        # Query without semantic search - use iterator to get all results, then sort and limit
        self.load_collection()

        filters = [f"timestamp >= {start_ts}", f"timestamp <= {end_ts}"]
        if chat_id:
            filters.append(f'chat_id == "{chat_id}"')

        expr = " and ".join(filters)

        if self._collection is None:
            return []

        # Use query_iterator to get ALL matching results, then sort and limit
        all_results: list[dict[str, Any]] = []
        iterator = self._collection.query_iterator(
            expr=expr,
            output_fields=["content", "sender", "timestamp", "chat_id", "chat_name"],
            batch_size=1000,
        )
        
        while True:
            batch = iterator.next()
            if not batch:
                break
            all_results.extend(batch)
        
        iterator.close()

        # Sort by timestamp (chronological order) FIRST, then apply limit
        sorted_results = sorted(all_results, key=lambda r: r.get("timestamp", 0))
        limited_results = sorted_results[:limit]

        return [
            {
                "content": r.get("content"),
                "sender": r.get("sender"),
                "timestamp": r.get("timestamp"),
                "chat_id": r.get("chat_id"),
                "chat_name": r.get("chat_name"),
                "score": None,
            }
            for r in limited_results
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
            List of messages from the specified sender, sorted chronologically.
        """
        if query_embedding:
            # For semantic search, we need to get more results, sort, then limit
            raw_results = self.search(
                query_embedding=query_embedding,
                limit=min(limit * 10, 5000),  # Cap at 5000 to avoid performance issues
                chat_id=chat_id,
                sender=sender,
            )
            # Sort by timestamp and apply limit
            sorted_results = sorted(raw_results, key=lambda r: r.get("timestamp", 0))
            return sorted_results[:limit]

        # Query without semantic search - use iterator to get all results, then sort and limit
        self.load_collection()

        filters = [f'sender == "{sender}"']
        if chat_id:
            filters.append(f'chat_id == "{chat_id}"')

        expr = " and ".join(filters)

        if self._collection is None:
            return []

        # Use query_iterator to get ALL matching results, then sort and limit
        all_results: list[dict[str, Any]] = []
        iterator = self._collection.query_iterator(
            expr=expr,
            output_fields=["content", "sender", "timestamp", "chat_id", "chat_name"],
            batch_size=1000,
        )
        
        while True:
            batch = iterator.next()
            if not batch:
                break
            all_results.extend(batch)
        
        iterator.close()

        # Sort by timestamp (chronological order) FIRST, then apply limit
        sorted_results = sorted(all_results, key=lambda r: r.get("timestamp", 0))
        limited_results = sorted_results[:limit]

        return [
            {
                "content": r.get("content"),
                "sender": r.get("sender"),
                "timestamp": r.get("timestamp"),
                "chat_id": r.get("chat_id"),
                "chat_name": r.get("chat_name"),
                "score": None,
            }
            for r in limited_results
        ]

    def list_chats(self) -> list[dict[str, Any]]:
        """List all available chats with statistics.

        Returns:
            List of chat information including message counts and date ranges.
        """
        self.load_collection()

        if self._collection is None:
            return []

        # Use query iterator to handle large collections efficiently
        # First, collect all unique chat_ids and their names using iterator
        chat_ids = set()
        chat_names: dict[str, str] = {}
        
        # Use query_iterator to bypass the 16384 limit
        iterator = self._collection.query_iterator(
            expr="",
            output_fields=["chat_id", "chat_name"],
            batch_size=5000,
        )
        
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
                
            for r in batch_results:
                chat_id = r.get("chat_id")
                chat_ids.add(chat_id)
                if chat_id not in chat_names:
                    chat_names[chat_id] = r.get("chat_name", "")
        
        iterator.close()

        # For each chat_id, get statistics using query iterator
        chats: list[dict[str, Any]] = []
        for chat_id in chat_ids:
            participants: set[str] = set()
            timestamps: list[int] = []
            total_count = 0
            
            # Use query_iterator for each chat
            chat_iterator = self._collection.query_iterator(
                expr=f'chat_id == "{chat_id}"',
                output_fields=["sender", "timestamp"],
                batch_size=5000,
            )
            
            while True:
                chat_results = chat_iterator.next()
                if not chat_results:
                    break
                
                total_count += len(chat_results)
                
                for r in chat_results:
                    participants.add(r.get("sender", ""))
                    timestamps.append(r.get("timestamp", 0))
            
            chat_iterator.close()

            if timestamps:
                chats.append({
                    "chat_id": chat_id,
                    "chat_name": chat_names.get(chat_id, ""),
                    "message_count": total_count,
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
