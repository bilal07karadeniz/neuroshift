"""
EMBEDDERS
==========
The store needs something that turns text into float vectors. Semantics come
entirely from the embedding model; the binary index only makes storing and
searching those vectors cheap.
"""

from typing import Callable, Optional, Sequence

import numpy as np


class Embedder:
    """Base class. Subclasses implement embed(); embed_queries() defaults to it.

    `name` is saved with an index and checked on load, so an index is never
    queried with vectors from a different model.
    """

    name: str = "custom"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents. Returns float32 array of shape (len(texts), dim)."""
        raise NotImplementedError

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed search queries. Override for models with query-specific prompts."""
        return self.embed(texts)


class FunctionEmbedder(Embedder):
    """Wraps any callable mapping a list of strings to a 2-D array."""

    def __init__(self, fn: Callable[[Sequence[str]], np.ndarray], name: str = "custom",
                 query_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None):
        self.fn = fn
        self.query_fn = query_fn
        self.name = name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.fn(list(texts)), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_fn is None:
            return self.embed(texts)
        return np.asarray(self.query_fn(list(texts)), dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a sentence-transformers model.

    Requires the optional dependency: pip install "neuroshift[embed]"

    Some models expect instruction prefixes (e.g. E5 uses "query: " and
    "passage: "); pass them as query_prefix / document_prefix.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None, batch_size: int = 64,
                 query_prefix: str = "", document_prefix: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("SentenceTransformerEmbedder requires sentence-transformers: "
                              "pip install \"neuroshift[embed]\"") from e
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])
