"""
HAMMING DISTANCE KERNELS
=========================
XOR + popcount over packed bit codes.

Codes are scanned in "word-major" layout: an array of shape (n_words, n_docs)
of uint64, so each pass XORs one 64-bit word of every document against the
same query word, and accumulates into a small integer buffer. This avoids
materializing (n_docs, n_bytes) temporaries and is several times faster than a
row-by-row scan in numpy.

With numpy >= 2.0 popcount is the native np.bitwise_count; older numpy falls
back to a byte lookup table.
"""

import numpy as np


WORD_BYTES = 8
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
HAS_NATIVE_POPCOUNT = hasattr(np, "bitwise_count")


def n_words(n_bytes: int) -> int:
    return -(-n_bytes // WORD_BYTES)


def to_words(codes: np.ndarray) -> np.ndarray:
    """Row-major packed codes (n, n_bytes) uint8 -> word-major (n_words, n) uint64.

    Codes are zero-padded to whole 64-bit words; padding bits are zero in every
    code, so they never contribute to a distance.
    """
    codes = np.atleast_2d(np.asarray(codes, dtype=np.uint8))
    n, b = codes.shape
    padded = np.zeros((n, n_words(b) * WORD_BYTES), dtype=np.uint8)
    padded[:, :b] = codes
    return np.ascontiguousarray(padded.view(np.uint64).T)


def from_words(words: np.ndarray, n_bytes: int) -> np.ndarray:
    """Inverse of to_words: (n_words, n) uint64 -> (n, n_bytes) uint8."""
    rows = np.ascontiguousarray(words.T).view(np.uint8)
    return np.ascontiguousarray(rows[:, :n_bytes])


def popcount(x: np.ndarray, native: bool = HAS_NATIVE_POPCOUNT) -> np.ndarray:
    """Per-element number of set bits of a uint64 array, as uint8."""
    if native:
        return np.bitwise_count(x)
    return _POPCOUNT_LUT[x.view(np.uint8)].reshape(*x.shape, WORD_BYTES).sum(axis=-1, dtype=np.uint8)


def hamming_words(query_words: np.ndarray, words: np.ndarray, chunk_size: int = 16384,
                  native: bool = HAS_NATIVE_POPCOUNT) -> np.ndarray:
    """Distances between queries (n_queries, n_words) and codes (n_words, n), both uint64.

    Returns (n_queries, n) uint16 (uint32 for codes wider than 65535 bits).
    """
    query_words = np.atleast_2d(query_words)
    nq, w = query_words.shape
    if words.shape[0] != w:
        raise ValueError(f"code width mismatch: queries have {w} words, codes have {words.shape[0]}")
    n = words.shape[1]
    out = np.zeros((nq, n), dtype=np.uint16 if w * 64 < 2 ** 16 else np.uint32)
    tmp = np.empty((nq, min(chunk_size, n)), dtype=np.uint64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        t = tmp[:, :end - start]
        acc = out[:, start:end]
        for i in range(w):
            np.bitwise_xor(words[i, start:end][None, :], query_words[:, i:i + 1], out=t)
            acc += popcount(t, native)
    return out


def hamming_distances(queries: np.ndarray, codes: np.ndarray, chunk_size: int = 16384) -> np.ndarray:
    """Hamming distances between row-major packed codes: (q, B) x (n, B) -> (q, n).

    Convenience wrapper; indexes keep codes in word-major layout and call
    hamming_words directly to avoid the conversion.
    """
    queries = np.atleast_2d(queries)
    if queries.shape[1] != codes.shape[1]:
        raise ValueError(f"code width mismatch: queries have {queries.shape[1]} bytes, "
                         f"codes have {codes.shape[1]}")
    return hamming_words(to_words(queries).T, to_words(codes), chunk_size)


def top_k(scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Indices of the k best scores, best first. Ties among them are ordered by index."""
    scores = np.asarray(scores)
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    if scores.dtype.kind == "u":
        scores = scores.astype(np.int64)
    keyed = -scores if largest else scores
    if k < scores.size:
        idx = np.argpartition(keyed, k - 1)[:k]
    else:
        idx = np.arange(scores.size)
    return idx[np.lexsort((idx, keyed[idx]))]
