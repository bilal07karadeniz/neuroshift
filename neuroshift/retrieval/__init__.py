"""
Binary semantic retrieval: real embeddings, stored as bits, searched with XOR + popcount.
"""

from .embedders import Embedder, FunctionEmbedder, SentenceTransformerEmbedder
from .hamming import hamming_distances
from .index import BinaryIndex
from .quantize import BinaryQuantizer
from .store import BinaryVectorStore, SearchResult

__all__ = [
    "BinaryIndex",
    "BinaryQuantizer",
    "BinaryVectorStore",
    "Embedder",
    "FunctionEmbedder",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "hamming_distances",
]
