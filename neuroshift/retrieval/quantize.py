"""
BINARY QUANTIZATION OF EMBEDDINGS
==================================
Turns float embeddings into packed bit codes. Two methods:

  - "sign":    one bit per embedding dimension (bit = value > 0).
               384-dim float32 (1536 bytes) -> 48 bytes. 32x smaller.
  - "simhash": random Gaussian projection to n_bits, then sign.
               This is locality-sensitive hashing: the fraction of differing
               bits estimates the angle between two vectors, so more bits buy
               more accuracy. Lets you trade memory for recall.

Centering (subtracting the corpus mean from documents and queries before
taking signs) is optional. It is essential for models whose embeddings carry a
large constant offset, whose bits would otherwise be nearly identical for every
document. Well-behaved models like all-MiniLM-L6-v2 do as well or better
without it; see benchmarks/bench_retrieval.py.
"""

from typing import Dict, Optional

import numpy as np


METHODS = ("sign", "simhash")


def as_matrix(x, dim: Optional[int] = None) -> np.ndarray:
    """Coerce input to a 2-D float32 array, checking the embedding width."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array of embeddings, got shape {x.shape}")
    if dim is not None and x.shape[1] != dim:
        raise ValueError(f"expected embeddings of width {dim}, got {x.shape[1]}")
    return x


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class BinaryQuantizer:
    """Maps float embeddings to packed binary codes (uint8, n_bits // 8 bytes per row)."""

    def __init__(self, dim: int, n_bits: Optional[int] = None, method: str = "sign",
                 center: bool = False, seed: int = 0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if n_bits is None:
            n_bits = dim
        if method == "sign" and n_bits != dim:
            raise ValueError(f"method='sign' produces one bit per dimension (n_bits={dim}); "
                             f"use method='simhash' for a different bit count")
        if n_bits <= 0 or n_bits % 8:
            raise ValueError(f"n_bits must be a positive multiple of 8, got {n_bits}")

        self.dim = dim
        self.n_bits = n_bits
        self.method = method
        self.center = center
        self.seed = seed

        self.mean: Optional[np.ndarray] = None if center else np.zeros(dim, dtype=np.float32)
        self.projection: Optional[np.ndarray] = None
        if method == "simhash":
            rng = np.random.default_rng(seed)
            self.projection = rng.standard_normal((dim, n_bits)).astype(np.float32)

    @property
    def n_bytes(self) -> int:
        return self.n_bits // 8

    @property
    def fitted(self) -> bool:
        return self.mean is not None

    def fit(self, embeddings) -> "BinaryQuantizer":
        """Estimate the centering offset from a representative sample of embeddings."""
        x = l2_normalize(as_matrix(embeddings, self.dim))
        self.mean = x.mean(axis=0) if self.center else np.zeros(self.dim, dtype=np.float32)
        return self

    def transform(self, embeddings) -> np.ndarray:
        """Float values whose signs become the bits, shape (n, n_bits).

        Used directly for asymmetric rescoring: the query keeps full precision
        while documents are only stored as bits.
        """
        if not self.fitted:
            raise RuntimeError("quantizer is not fitted: call fit() with a sample of embeddings "
                               "first, or construct it with center=False")
        x = l2_normalize(as_matrix(embeddings, self.dim)) - self.mean
        if self.projection is not None:
            x = x @ self.projection
        return x

    def encode(self, embeddings) -> np.ndarray:
        """Packed binary codes, shape (n, n_bits // 8), dtype uint8."""
        return np.packbits(self.transform(embeddings) > 0, axis=1)

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {
            "dim": np.array(self.dim),
            "n_bits": np.array(self.n_bits),
            "method": np.array(self.method),
            "center": np.array(self.center),
            "seed": np.array(self.seed),
        }
        if self.mean is not None:
            state["mean"] = self.mean
        if self.projection is not None:
            # Stored rather than regenerated from the seed: numpy does not
            # guarantee identical random streams across versions.
            state["projection"] = self.projection
        return state

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "BinaryQuantizer":
        q = cls(dim=int(state["dim"]), n_bits=int(state["n_bits"]), method=str(state["method"]),
                center=bool(state["center"]), seed=int(state["seed"]))
        if "mean" in state:
            q.mean = np.asarray(state["mean"], dtype=np.float32)
        if "projection" in state:
            q.projection = np.asarray(state["projection"], dtype=np.float32)
        return q
