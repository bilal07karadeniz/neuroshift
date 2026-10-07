"""
BINARY VECTOR STORE
====================
A document store for RAG and semantic search that keeps 1 bit per embedding
dimension instead of a 32-bit float:

    store = BinaryVectorStore(SentenceTransformerEmbedder())
    store.add(docs, tags=[["python", "web"], ...])
    store.search("how do I build an API?", k=5, exclude_tags=["deprecated"])
    store.save("kb.npz")

  - Semantic: meaning comes from a real embedding model.
  - Small:    "sign" codes are 32x smaller than float32 vectors.
  - Simple:   one .npz file, numpy only at query time, no server.
  - Safe:     files load with allow_pickle=False.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .embedders import Embedder
from .index import BinaryIndex


FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    doc_id: int
    score: float
    text: str
    metadata: Dict = field(default_factory=dict)
    tags: Tuple[str, ...] = ()


def _norm_tag(tag: str) -> str:
    return tag.strip().lower()


class BinaryVectorStore:
    """Texts + metadata + tags over a BinaryIndex, with tag filters and concept algebra."""

    def __init__(self, embedder: Embedder, method: str = "sign", n_bits: Optional[int] = None,
                 center: bool = False, seed: int = 0, rescore_multiplier: int = 10,
                 min_fit_size: int = 16):
        self.embedder = embedder
        self._index_config = dict(n_bits=n_bits, method=method, center=center, seed=seed,
                                  rescore_multiplier=rescore_multiplier, min_fit_size=min_fit_size)
        self.index: Optional[BinaryIndex] = None
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self.tags: List[Tuple[str, ...]] = []
        self._tag_index: Dict[str, Set[int]] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def _ensure_index(self, dim: int) -> BinaryIndex:
        if self.index is None:
            self.index = BinaryIndex(dim, **self._index_config)
        return self.index

    def fit(self, texts: Sequence[str]) -> "BinaryVectorStore":
        """Fit the centering offset on representative texts before the first add().

        Only needed when the first add() batch is small; otherwise the first
        batch is used automatically.
        """
        vectors = self.embedder.embed(list(texts))
        self._ensure_index(vectors.shape[1]).fit(vectors)
        return self

    def add(self, texts: Union[str, Sequence[str]], metadata: Optional[Sequence[Dict]] = None,
            tags: Optional[Sequence[Iterable[str]]] = None) -> List[int]:
        """Embed and index documents. Returns their doc ids."""
        if isinstance(texts, str):
            texts = [texts]
            metadata = [metadata] if isinstance(metadata, dict) else metadata
            if tags is not None and all(isinstance(t, str) for t in tags):
                tags = [tags]
        texts = list(texts)
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(f"got {len(metadata)} metadata entries for {len(texts)} texts")
        if tags is not None and len(tags) != len(texts):
            raise ValueError(f"got {len(tags)} tag lists for {len(texts)} texts")
        if not texts:
            return []

        vectors = self.embedder.embed(texts)
        ids = self._ensure_index(vectors.shape[1]).add(vectors)

        for i, doc_id in enumerate(ids.tolist()):
            doc_tags = ()
            if tags is not None:
                raw = [tags[i]] if isinstance(tags[i], str) else tags[i]
                doc_tags = tuple(sorted({_norm_tag(t) for t in raw}))
            self.texts.append(texts[i])
            self.metadata.append(dict(metadata[i]) if metadata is not None else {})
            self.tags.append(doc_tags)
            for t in doc_tags:
                self._tag_index.setdefault(t, set()).add(doc_id)
        return ids.tolist()

    def search(self, query: str, k: int = 5, *, all_tags: Sequence[str] = (),
               any_tags: Sequence[str] = (), exclude_tags: Sequence[str] = (),
               rescore: bool = True) -> List[SearchResult]:
        """Semantic search, optionally restricted by tags.

        Args:
            all_tags: documents must carry every one of these tags.
            any_tags: documents must carry at least one of these tags.
            exclude_tags: documents carrying any of these tags are dropped.
        """
        return self.search_composite([query], k=k, all_tags=all_tags, any_tags=any_tags,
                                     exclude_tags=exclude_tags, rescore=rescore)

    def search_composite(self, positive: Sequence[str], negative: Sequence[str] = (), k: int = 5,
                         negative_weight: float = 0.5, *, all_tags: Sequence[str] = (),
                         any_tags: Sequence[str] = (), exclude_tags: Sequence[str] = (),
                         rescore: bool = True) -> List[SearchResult]:
        """Concept algebra: documents about ALL positive concepts, demoted if about ANY negative one.

        Example: search_composite(["python", "web framework"], negative=["django"])
        """
        if not self.texts or not positive:
            return []
        allowed = self._filter_ids(all_tags, any_tags, exclude_tags)
        pos = self.embedder.embed_queries(list(positive))
        neg = self.embedder.embed_queries(list(negative)) if negative else None
        ids, scores = self.index.search_composite(pos, neg, k=k, negative_weight=negative_weight,
                                                  rescore=rescore, ids=allowed)
        return [self._result(i, s) for i, s in zip(ids.tolist(), scores.tolist())]

    def _filter_ids(self, all_tags: Sequence[str], any_tags: Sequence[str],
                    exclude_tags: Sequence[str]) -> Optional[np.ndarray]:
        if not (all_tags or any_tags or exclude_tags):
            return None
        allowed = set(range(len(self.texts)))
        for t in all_tags:
            allowed &= self._tag_index.get(_norm_tag(t), set())
        if any_tags:
            allowed &= set().union(*(self._tag_index.get(_norm_tag(t), set()) for t in any_tags))
        for t in exclude_tags:
            allowed -= self._tag_index.get(_norm_tag(t), set())
        return np.array(sorted(allowed), dtype=np.int64)

    def _result(self, doc_id: int, score: float) -> SearchResult:
        return SearchResult(doc_id=doc_id, score=score, text=self.texts[doc_id],
                            metadata=self.metadata[doc_id], tags=self.tags[doc_id])

    def stats(self) -> Dict:
        if self.index is None:
            return {"documents": 0}
        bytes_per_doc = self.index.quantizer.n_bytes
        float_bytes_per_doc = self.index.dim * 4
        return {
            "documents": len(self),
            "embedding_dim": self.index.dim,
            "method": self.index.quantizer.method,
            "bits_per_doc": self.index.n_bits,
            "bytes_per_doc": bytes_per_doc,
            "index_bytes": self.index.memory_bytes(),
            "float32_equivalent_bytes": len(self) * float_bytes_per_doc,
            "compression_vs_float32": float_bytes_per_doc / bytes_per_doc,
            "tags": sorted(self._tag_index),
        }

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write the store to a single .npz file. Returns the path written.

        Metadata must be JSON-serializable. Embeddings are not stored, only bit codes.
        """
        if self.index is None:
            raise ValueError("nothing to save: the store is empty")
        path = os.fspath(path)
        if not path.endswith(".npz"):
            path += ".npz"
        payload = {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder.name,
            "texts": self.texts,
            "metadata": self.metadata,
            "tags": [list(t) for t in self.tags],
        }
        try:
            blob = json.dumps(payload).encode("utf-8")
        except TypeError as e:
            raise TypeError(f"document metadata must be JSON-serializable to save: {e}") from e
        arrays = self.index.state_dict()
        arrays["store"] = np.frombuffer(blob, dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Union[str, os.PathLike], embedder: Embedder,
             check_embedder: bool = True) -> "BinaryVectorStore":
        """Load a store saved with save().

        Raises ValueError if `embedder` differs from the one the store was built
        with, since mixing models silently returns meaningless results.
        """
        with np.load(os.fspath(path), allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        payload = json.loads(arrays.pop("store").tobytes().decode("utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported store format version {payload.get('format_version')!r}")
        if check_embedder and payload["embedder"] != embedder.name:
            raise ValueError(f"store was built with embedder {payload['embedder']!r} but "
                             f"{embedder.name!r} was given; pass check_embedder=False to override")

        index = BinaryIndex.from_state_dict(arrays)
        store = cls(embedder, method=index.quantizer.method, n_bits=index.n_bits,
                    center=index.quantizer.center, seed=index.quantizer.seed,
                    rescore_multiplier=index.rescore_multiplier, min_fit_size=index.min_fit_size)
        store.index = index
        store.texts = list(payload["texts"])
        store.metadata = [dict(m) for m in payload["metadata"]]
        store.tags = [tuple(t) for t in payload["tags"]]
        for doc_id, doc_tags in enumerate(store.tags):
            for t in doc_tags:
                store._tag_index.setdefault(t, set()).add(doc_id)
        if len(store.texts) != len(index):
            raise ValueError(f"corrupt store: {len(store.texts)} texts but {len(index)} codes")
        return store
