"""Embedding service for generating text embeddings.

This module provides a service for generating semantic embeddings using
sentence-transformers models. It supports lazy loading and batch processing
for efficient embedding generation.
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model for multilingual support
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Embedding dimension for the default model
DEFAULT_DIMENSION = 768


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers.

    This service provides lazy loading of the embedding model (loaded on first use)
    and supports batch processing for efficient embedding generation.

    Attributes:
        model_name: The name of the sentence-transformers model to use.
        dimension: The dimension of the embedding vectors.

    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.encode("Hello, world!")
        >>> len(embedding)
        768
        >>> embeddings = service.encode_batch(["Hello", "World"])
        >>> len(embeddings)
        2
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedding service.

        Args:
            model_name: The name of the sentence-transformers model to use.
                       Defaults to the EMBEDDING_MODEL environment variable
                       or 'paraphrase-multilingual-mpnet-base-v2'.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self._model: "SentenceTransformer | None" = None
        self._dimension: int | None = None

    @property
    def model(self) -> "SentenceTransformer":
        """Get the sentence-transformers model, loading it if necessary.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for embedding generation. "
                    "Install it with: pip install sentence-transformers"
                ) from e

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded embedding model: %s (dimension: %d)",
                self.model_name,
                self._dimension,
            )

        return self._model

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            The dimension of the embedding vectors.
        """
        if self._dimension is None:
            # Trigger model loading to get the dimension
            _ = self.model
        return self._dimension or DEFAULT_DIMENSION

    def encode(self, text: str, normalize: bool = True) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to encode.
            normalize: Whether to normalize the embedding vector.

        Returns:
            A list of floats representing the embedding vector.

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.encode("Hello, world!")
            >>> isinstance(embedding, list)
            True
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        This method processes texts in batches for memory efficiency.

        Args:
            texts: List of texts to encode.
            batch_size: Number of texts to process in each batch.
            normalize: Whether to normalize the embedding vectors.
            show_progress: Whether to show a progress bar.

        Returns:
            A list of embedding vectors, one for each input text.

        Example:
            >>> service = EmbeddingService()
            >>> embeddings = service.encode_batch(["Hello", "World"])
            >>> len(embeddings)
            2
            >>> len(embeddings[0])
            768
        """
        if not texts:
            return []

        logger.debug("Encoding %d texts with batch size %d", len(texts), batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

        return embeddings.tolist()


# Global singleton instance for convenience
_default_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the default embedding service instance.

    This function returns a singleton instance of the EmbeddingService,
    creating it if necessary.

    Returns:
        The default EmbeddingService instance.

    Example:
        >>> service = get_embedding_service()
        >>> isinstance(service, EmbeddingService)
        True
    """
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def encode_text(text: str) -> list[float]:
    """Convenience function to encode a single text.

    Args:
        text: The text to encode.

    Returns:
        The embedding vector as a list of floats.

    Example:
        >>> embedding = encode_text("Hello, world!")
        >>> len(embedding)
        768
    """
    return get_embedding_service().encode(text)


def encode_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Convenience function to encode multiple texts.

    Args:
        texts: List of texts to encode.
        batch_size: Number of texts to process in each batch.

    Returns:
        List of embedding vectors.

    Example:
        >>> embeddings = encode_texts(["Hello", "World"])
        >>> len(embeddings)
        2
    """
    return get_embedding_service().encode_batch(texts, batch_size=batch_size)
