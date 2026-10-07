"""
BINARY VECTOR INDEX
====================
Stores only packed bit codes and searches them in two stages:

  1. Shortlist: Hamming distance (XOR + popcount) over every stored code.
  2. Rescore:   the full-precision query is scored against the shortlisted
                codes' bits ("asymmetric" scoring). This recovers most of the
                ranking quality lost to binarization without storing any
                float vectors.

Scores are similarity estimates in [-1, 1]: comparable within one index,
but not equal to the true cosine similarity.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .hamming import from_words, hamming_words, n_words, to_words, top_k
from .quantize import BinaryQuantizer, as_matrix


class BinaryIndex:
    """Search index over binary-quantized embeddings."""

    def __init__(self, dim: int, n_bits: Optional[int] = None, method: str = "sign",
                 center: bool = False, seed: int = 0, rescore_multiplier: int = 10,
                 min_fit_size: int = 16):
        self.quantizer = BinaryQuantizer(dim, n_bits=n_bits, method=method, center=center, seed=seed)
        self.rescore_multiplier = rescore_multiplier
        self.min_fit_size = min_fit_size
        # Word-major storage (n_words, capacity) for fast scans; see hamming.py.
        self._words = np.empty((n_words(self.quantizer.n_bytes), 64), dtype=np.uint64)
        self._size = 0

    @property
    def dim(self) -> int:
        return self.quantizer.dim

    @property
    def n_bits(self) -> int:
        return self.quantizer.n_bits

    @property
    def words(self) -> np.ndarray:
        """Stored codes in word-major layout, shape (n_words, len(self)), uint64."""
        return self._words[:, :self._size]

    @property
    def codes(self) -> np.ndarray:
        """Stored codes in row-major packed layout, shape (len(self), n_bits // 8), uint8."""
        return from_words(self.words, self.quantizer.n_bytes)

    def __len__(self) -> int:
        return self._size

    def fit(self, sample) -> "BinaryIndex":
        """Fit the centering offset on representative embeddings. Must precede add()."""
        if self._size:
            raise RuntimeError("cannot refit after codes were added: existing codes would no "
                               "longer match new ones")
        self.quantizer.fit(sample)
        return self

    def add(self, embeddings) -> np.ndarray:
        """Quantize and store embeddings. Returns their ids (consecutive integers)."""
        x = as_matrix(embeddings, self.dim)
        if not self.quantizer.fitted:
            if len(x) < self.min_fit_size:
                raise ValueError(
                    f"the first batch has {len(x)} embeddings, too few to estimate the centering "
                    f"offset (need {self.min_fit_size}). Call fit() with a representative sample "
                    f"first, add a larger first batch, or use center=False.")
            self.quantizer.fit(x)

        new_words = to_words(self.quantizer.encode(x))
        needed = self._size + new_words.shape[1]
        if needed > self._words.shape[1]:
            grown = np.empty((self._words.shape[0], max(needed, 2 * self._words.shape[1])), dtype=np.uint64)
            grown[:, :self._size] = self.words
            self._words = grown
        self._words[:, self._size:needed] = new_words
        ids = np.arange(self._size, needed)
        self._size = needed
        return ids

    def search(self, queries, k: int = 10, rescore: bool = True,
               ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest neighbours for each query.

        Args:
            queries: (q, dim) or (dim,) float embeddings.
            k: results per query.
            rescore: rerank the Hamming shortlist with full-precision queries.
            ids: restrict the search to these document ids (e.g. a tag filter).

        Returns:
            (ids, scores), both of shape (q, min(k, n_candidates)), best first.
        """
        q = as_matrix(queries, self.dim)
        results = [self.search_composite(row, k=k, rescore=rescore, ids=ids) for row in q]
        width = min(k, len(self) if ids is None else len(ids))
        out_ids = np.array([r[0] for r in results], dtype=np.int64).reshape(len(q), width)
        out_scores = np.array([r[1] for r in results], dtype=np.float32).reshape(len(q), width)
        return out_ids, out_scores

    def search_composite(self, positive, negative=None, k: int = 10,
                         negative_weight: float = 0.5, rescore: bool = True,
                         ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Concept-algebra search.

        score(doc) = min(sim to each positive) - negative_weight * max(sim to each negative)

        A document must match every positive concept to score well, and is
        demoted if it matches any negative one.

        Returns:
            (ids, scores) of shape (min(k, n_candidates),), best first.
        """
        if not self.quantizer.fitted:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        pos = as_matrix(positive, self.dim)
        neg = as_matrix(negative, self.dim) if negative is not None and len(negative) else None
        if ids is None:
            candidates, words = np.arange(self._size), self.words
        else:
            candidates = np.asarray(ids, dtype=np.int64)
            words = self.words[:, candidates]
        if candidates.size == 0 or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        # Stage 1: Hamming shortlist. cos(pi * h / n_bits) is the SimHash angle
        # estimate; for a single plain query, rank by raw distance and skip it.
        width = k if not rescore else k * self.rescore_multiplier
        concepts = pos if neg is None else np.concatenate([pos, neg])
        dist = hamming_words(to_words(self.quantizer.encode(concepts)).T, words)
        if len(concepts) == 1:
            shortlist = top_k(dist[0], width, largest=False)
            stage1 = self._hamming_similarity(dist[:, shortlist])[0]
        else:
            sims = self._hamming_similarity(dist)
            scores = self._combine(sims[:len(pos)], sims[len(pos):] if neg is not None else None,
                                   negative_weight)
            shortlist = top_k(scores, width)
            stage1 = scores[shortlist]
        if not rescore:
            return candidates[shortlist], stage1.astype(np.float32)

        # Stage 2: asymmetric rescoring of the shortlist.
        short_codes = from_words(words[:, shortlist], self.quantizer.n_bytes)
        sims = self._asymmetric_similarity(concepts, short_codes)
        stage2 = self._combine(sims[:len(pos)], sims[len(pos):] if neg is not None else None,
                               negative_weight)
        best = top_k(stage2, k)
        return candidates[shortlist[best]], stage2[best].astype(np.float32)

    def _hamming_similarity(self, dist: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * dist / self.n_bits)

    def _asymmetric_similarity(self, embeddings: np.ndarray, codes: np.ndarray) -> np.ndarray:
        t = self.quantizer.transform(embeddings)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        signs = np.unpackbits(codes, axis=1, count=self.n_bits).astype(np.float32) * 2 - 1
        return (t @ signs.T) / np.sqrt(self.n_bits)

    @staticmethod
    def _combine(pos_sims: np.ndarray, neg_sims: Optional[np.ndarray],
                 negative_weight: float) -> np.ndarray:
        score = pos_sims.min(axis=0)
        if neg_sims is not None:
            score = score - negative_weight * neg_sims.max(axis=0)
        return score

    def memory_bytes(self) -> int:
        """Bytes used by the stored codes (including padding to whole 64-bit words)."""
        return self.words.nbytes

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {f"quantizer.{k}": v for k, v in self.quantizer.state_dict().items()}
        state["codes"] = self.codes  # row-major uint8: byte-order independent on disk
        state["rescore_multiplier"] = np.array(self.rescore_multiplier)
        state["min_fit_size"] = np.array(self.min_fit_size)
        return state

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "BinaryIndex":
        qstate = {k[len("quantizer."):]: v for k, v in state.items() if k.startswith("quantizer.")}
        index = cls.__new__(cls)
        index.quantizer = BinaryQuantizer.from_state_dict(qstate)
        index.rescore_multiplier = int(state["rescore_multiplier"])
        index.min_fit_size = int(state["min_fit_size"])
        codes = np.asarray(state["codes"], dtype=np.uint8)
        if codes.ndim != 2 or codes.shape[1] != index.quantizer.n_bytes:
            raise ValueError(f"stored codes have shape {codes.shape}, expected (n, {index.quantizer.n_bytes})")
        index._words = to_words(codes)
        index._size = len(codes)
        return index
