"""Tests for neuroshift.retrieval (binary quantization, Hamming search, vector store)."""

import hashlib

import numpy as np
import pytest

from neuroshift.retrieval import (
    BinaryIndex,
    BinaryQuantizer,
    BinaryVectorStore,
    FunctionEmbedder,
    hamming_distances,
)
from neuroshift.retrieval.hamming import from_words, hamming_words, popcount, to_words, top_k


DIM = 64


def _word_vec(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def _toy_embed(texts):
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    A constant offset mimics the mild anisotropy of real embedding models
    (normalized all-MiniLM-L6-v2 vectors have a mean of norm ~0.36).
    """
    offset = np.full(DIM, 0.05, dtype=np.float32)
    rows = []
    for text in texts:
        words = text.lower().split()
        v = sum((_word_vec(w) for w in words), np.zeros(DIM, dtype=np.float32))
        rows.append(v / max(np.linalg.norm(v), 1e-6) + offset)
    return np.stack(rows)


@pytest.fixture
def embedder():
    return FunctionEmbedder(_toy_embed, name="toy-bow")


@pytest.fixture
def clustered():
    """200 points in 10 tight clusters, plus one query per cluster."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((10, DIM)).astype(np.float32)
    labels = np.repeat(np.arange(10), 20)
    points = centers[labels] + 0.15 * rng.standard_normal((200, DIM)).astype(np.float32)
    queries = centers + 0.15 * rng.standard_normal((10, DIM)).astype(np.float32)
    return points, labels, queries


def _exact_top_k(points, queries, k):
    p = points / np.linalg.norm(points, axis=1, keepdims=True)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return np.argsort(-(q @ p.T), axis=1)[:, :k]


DOCS = [
    ("python web framework for building apis", ["python", "web"]),
    ("django python web framework with an orm", ["python", "web", "django"]),
    ("flask python micro web framework", ["python", "web"]),
    ("javascript library for browser user interfaces", ["javascript", "web"]),
    ("postgres relational database with sql", ["database"]),
    ("redis in memory key value database", ["database"]),
    ("docker containers for deploying applications", ["devops"]),
    ("kubernetes orchestrates docker containers", ["devops"]),
    ("pandas dataframes for python data analysis", ["python", "data"]),
    ("numpy arrays for python numerical computing", ["python", "data"]),
    ("rust systems programming without garbage collection", ["rust"]),
    ("go language for concurrent network services", ["go"]),
    ("sql query optimization and indexes", ["database"]),
    ("react components for javascript web apps", ["javascript", "web"]),
    ("terraform infrastructure as code", ["devops"]),
    ("pytorch deep learning in python", ["python", "ml"]),
]


@pytest.fixture
def store(embedder):
    s = BinaryVectorStore(embedder)
    s.add([d for d, _ in DOCS], metadata=[{"i": i} for i in range(len(DOCS))],
          tags=[t for _, t in DOCS])
    return s


# -- quantizer ---------------------------------------------------------------

class TestBinaryQuantizer:

    def test_sign_codes_have_one_bit_per_dimension(self):
        q = BinaryQuantizer(DIM, center=False)
        codes = q.encode(np.random.default_rng(0).standard_normal((5, DIM)))
        assert codes.shape == (5, DIM // 8)
        assert codes.dtype == np.uint8

    def test_sign_bits_match_signs(self):
        x = np.random.default_rng(1).standard_normal((3, DIM)).astype(np.float32)
        q = BinaryQuantizer(DIM, center=False)
        bits = np.unpackbits(q.encode(x), axis=1)
        np.testing.assert_array_equal(bits, (x > 0).astype(np.uint8))

    def test_simhash_bit_count(self):
        q = BinaryQuantizer(DIM, n_bits=256, method="simhash", center=False)
        assert q.encode(np.ones((2, DIM))).shape == (2, 32)

    def test_sign_rejects_other_bit_counts(self):
        with pytest.raises(ValueError, match="simhash"):
            BinaryQuantizer(DIM, n_bits=128, method="sign")

    def test_rejects_bits_not_multiple_of_8(self):
        with pytest.raises(ValueError, match="multiple of 8"):
            BinaryQuantizer(DIM, n_bits=100, method="simhash")

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            BinaryQuantizer(DIM, method="pq")

    def test_encode_before_fit_raises_when_centering(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            BinaryQuantizer(DIM, center=True).encode(np.ones(DIM))

    def test_centering_removes_shared_offset(self):
        """With a large shared offset, uncentered sign codes are nearly identical; centered ones are not."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((100, DIM)).astype(np.float32) * 0.1 + 5.0
        raw = BinaryQuantizer(DIM, center=False).encode(x)
        centered = BinaryQuantizer(DIM, center=True).fit(x).encode(x)
        assert np.unpackbits(raw, axis=1).std(axis=0).mean() < 0.05
        assert np.unpackbits(centered, axis=1).std(axis=0).mean() > 0.4

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="width"):
            BinaryQuantizer(DIM, center=False).encode(np.ones((2, DIM + 1)))

    def test_state_dict_round_trip(self):
        x = np.random.default_rng(3).standard_normal((20, DIM))
        q = BinaryQuantizer(DIM, n_bits=128, method="simhash").fit(x)
        q2 = BinaryQuantizer.from_state_dict(q.state_dict())
        np.testing.assert_array_equal(q.encode(x), q2.encode(x))


# -- hamming -------------------------------------------------------------------

class TestHamming:

    @pytest.mark.parametrize("n_bytes", [3, 8, 48])
    def test_matches_naive_count(self, n_bytes):
        rng = np.random.default_rng(4)
        a = rng.integers(0, 256, (4, n_bytes), dtype=np.uint8)
        b = rng.integers(0, 256, (50, n_bytes), dtype=np.uint8)
        naive = (np.unpackbits(a, axis=1)[:, None, :] != np.unpackbits(b, axis=1)[None, :, :]).sum(-1)
        np.testing.assert_array_equal(hamming_distances(a, b), naive)

    def test_lookup_table_fallback_matches(self):
        x = np.random.default_rng(5).integers(0, 2 ** 63, (30, 4), dtype=np.uint64)
        expected = np.unpackbits(x.view(np.uint8), axis=1).reshape(30, 4, 64).sum(-1)
        np.testing.assert_array_equal(popcount(x, native=False), expected)
        words = to_words(np.random.default_rng(5).integers(0, 256, (40, 16), dtype=np.uint8))
        q = words[:, :3].T.copy()
        np.testing.assert_array_equal(hamming_words(q, words, native=False), hamming_words(q, words))

    @pytest.mark.parametrize("n_bytes", [3, 8, 13, 48])
    def test_word_layout_round_trip(self, n_bytes):
        codes = np.random.default_rng(13).integers(0, 256, (17, n_bytes), dtype=np.uint8)
        words = to_words(codes)
        assert words.shape == (-(-n_bytes // 8), 17) and words.dtype == np.uint64
        np.testing.assert_array_equal(from_words(words, n_bytes), codes)

    def test_chunking_does_not_change_result(self):
        rng = np.random.default_rng(6)
        a = rng.integers(0, 256, (2, 8), dtype=np.uint8)
        b = rng.integers(0, 256, (101, 8), dtype=np.uint8)
        np.testing.assert_array_equal(hamming_distances(a, b, chunk_size=7), hamming_distances(a, b))

    def test_width_mismatch_raises(self):
        with pytest.raises(ValueError, match="width"):
            hamming_distances(np.zeros((1, 4), np.uint8), np.zeros((3, 8), np.uint8))

    def test_top_k_orders_best_first(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        np.testing.assert_array_equal(top_k(scores, 2), [1, 3])
        np.testing.assert_array_equal(top_k(scores, 2, largest=False), [0, 2])
        np.testing.assert_array_equal(top_k(scores, 10), [1, 3, 2, 0])


# -- index ---------------------------------------------------------------------

class TestBinaryIndex:

    def test_finds_noisy_copy(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((500, DIM)).astype(np.float32)
        index = BinaryIndex(DIM)
        index.add(x)
        noisy = x[[3, 250, 499]] + 0.1 * rng.standard_normal((3, DIM)).astype(np.float32)
        ids, scores = index.search(noisy, k=1)
        np.testing.assert_array_equal(ids[:, 0], [3, 250, 499])
        assert scores.shape == (3, 1)

    @pytest.mark.parametrize("rescore", [True, False])
    def test_recall_against_exact_search(self, clustered, rescore):
        points, labels, queries = clustered
        index = BinaryIndex(DIM)
        index.add(points)
        ids, _ = index.search(queries, k=10, rescore=rescore)
        # Each query's 10 neighbours should all come from its own cluster.
        assert (labels[ids] == np.arange(10)[:, None]).mean() > 0.95

    def test_rescoring_does_not_hurt_overlap_with_exact(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal((2000, DIM)).astype(np.float32)
        q = rng.standard_normal((50, DIM)).astype(np.float32)
        exact = _exact_top_k(x, q, 10)
        index = BinaryIndex(DIM)
        index.add(x)

        def overlap(ids):
            return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])

        assert overlap(index.search(q, k=10, rescore=True)[0]) >= overlap(index.search(q, k=10, rescore=False)[0])

    def test_simhash_more_bits_improves_recall(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal((2000, DIM)).astype(np.float32)
        q = rng.standard_normal((50, DIM)).astype(np.float32)
        exact = _exact_top_k(x, q, 10)

        def recall(n_bits):
            index = BinaryIndex(DIM, n_bits=n_bits, method="simhash")
            index.add(x)
            ids, _ = index.search(q, k=10, rescore=False)
            return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])

        assert recall(1024) > recall(64)

    def test_centering_rescues_anisotropic_embeddings(self):
        """A large shared offset makes uncentered codes collapse; centering restores the neighbours."""
        rng = np.random.default_rng(15)
        x = rng.standard_normal((500, DIM)).astype(np.float32) + 4.0
        noisy = x[:50] + 0.1 * rng.standard_normal((50, DIM)).astype(np.float32)

        def hit_rate(center):
            index = BinaryIndex(DIM, center=center)
            index.add(x)
            return (index.search(noisy, k=1)[0][:, 0] == np.arange(50)).mean()

        assert hit_rate(center=True) > 0.9
        assert hit_rate(center=True) > hit_rate(center=False)

    def test_small_first_batch_raises_with_guidance(self):
        index = BinaryIndex(DIM, center=True)
        with pytest.raises(ValueError, match="fit"):
            index.add(np.ones((3, DIM)))

    def test_small_first_batch_ok_after_fit_or_without_centering(self):
        x = np.random.default_rng(10).standard_normal((40, DIM))
        BinaryIndex(DIM, center=True).fit(x).add(x[:2])
        BinaryIndex(DIM).add(x[:2])

    def test_refit_after_add_raises(self):
        x = np.random.default_rng(11).standard_normal((40, DIM))
        index = BinaryIndex(DIM, center=True)
        index.add(x)
        with pytest.raises(RuntimeError, match="refit"):
            index.fit(x)

    def test_incremental_adds_grow_buffer(self):
        x = np.random.default_rng(12).standard_normal((300, DIM)).astype(np.float32)
        index = BinaryIndex(DIM, center=True).fit(x)
        for row in x:
            index.add(row)
        assert len(index) == 300
        np.testing.assert_array_equal(index.codes, index.quantizer.encode(x))
        assert index.memory_bytes() == 300 * DIM // 8

    def test_search_restricted_to_ids(self, clustered):
        points, labels, queries = clustered
        index = BinaryIndex(DIM)
        index.add(points)
        allowed = np.flatnonzero(labels == 3)
        ids, _ = index.search(queries[0], k=5, ids=allowed)
        assert set(ids[0]) <= set(allowed.tolist())

    def test_k_larger_than_index(self, clustered):
        points, _, queries = clustered
        index = BinaryIndex(DIM)
        index.add(points[:20])
        ids, scores = index.search(queries[:2], k=50)
        assert ids.shape == scores.shape == (2, 20)

    def test_empty_index_returns_nothing(self):
        ids, scores = BinaryIndex(DIM).search(np.ones(DIM), k=5)
        assert ids.shape == (1, 0)

    def test_negative_concept_demotes_matches(self, clustered):
        points, labels, queries = clustered
        index = BinaryIndex(DIM)
        index.add(points)
        # Positive: midpoint of clusters 0 and 1. Negative: cluster 1.
        positive = queries[0] + queries[1]
        ids, _ = index.search_composite(positive, negative=queries[1:2], k=10, negative_weight=1.0)
        assert (labels[ids] == 0).mean() > (labels[ids] == 1).mean()

    def test_state_dict_round_trip(self, clustered):
        points, _, queries = clustered
        index = BinaryIndex(DIM, n_bits=128, method="simhash")
        index.add(points)
        restored = BinaryIndex.from_state_dict(index.state_dict())
        a, b = index.search(queries, k=5), restored.search(queries, k=5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


# -- store ---------------------------------------------------------------------

class TestBinaryVectorStore:

    def test_search_returns_results(self, store):
        results = store.search("python web framework", k=3)
        assert len(results) == 3
        assert {r.doc_id for r in results} == {0, 1, 2}
        for r in results:
            assert r.text == DOCS[r.doc_id][0]
            assert r.metadata == {"i": r.doc_id}
            assert set(r.tags) == set(DOCS[r.doc_id][1])
        assert results[0].score >= results[1].score >= results[2].score

    def test_all_tags_filter(self, store):
        results = store.search("framework", k=10, all_tags=["python", "web"])
        assert results and all({"python", "web"} <= set(r.tags) for r in results)

    def test_any_tags_filter(self, store):
        results = store.search("framework", k=20, any_tags=["rust", "go"])
        assert {r.text.split()[0] for r in results} == {"rust", "go"}

    def test_exclude_tags_filter(self, store):
        results = store.search("python web framework", k=20, exclude_tags=["Django"])
        assert results and all("django" not in r.tags for r in results)

    def test_filter_matching_nothing(self, store):
        assert store.search("python", all_tags=["nonexistent"]) == []

    def test_composite_negative_demotes(self, store):
        plain = [r.text for r in store.search_composite(["python web framework"], k=3)]
        assert any("django" in t for t in plain)
        filtered = [r.text for r in store.search_composite(["python web framework"], ["django orm"],
                                                           k=3, negative_weight=1.0)]
        assert not any("django" in t for t in filtered)

    def test_empty_store(self, embedder):
        s = BinaryVectorStore(embedder)
        assert s.search("anything") == []
        assert s.stats() == {"documents": 0}
        with pytest.raises(ValueError, match="empty"):
            s.save("unused.npz")

    def test_add_single_string_with_tags(self, embedder):
        s = BinaryVectorStore(embedder, center=False)
        s.add("hello world", metadata={"src": "a"}, tags=["greeting"])
        s.add(["second doc", "third doc"], tags=["x", ["y", "z"]])
        assert s.metadata[0] == {"src": "a"}
        assert s.tags == [("greeting",), ("x",), ("y", "z")]

    def test_mismatched_lengths_raise(self, embedder):
        s = BinaryVectorStore(embedder)
        with pytest.raises(ValueError, match="metadata"):
            s.add(["a", "b"], metadata=[{}])
        with pytest.raises(ValueError, match="tag"):
            s.add(["a", "b"], tags=[["x"]])

    def test_fit_then_small_batches(self, embedder):
        s = BinaryVectorStore(embedder, center=True)
        s.fit([d for d, _ in DOCS])
        s.add("rust systems programming")
        assert s.search("rust", k=1)[0].text == "rust systems programming"

    def test_stats_report_32x_compression(self, store):
        stats = store.stats()
        assert stats["documents"] == len(DOCS)
        assert stats["bytes_per_doc"] == DIM // 8
        assert stats["index_bytes"] == len(DOCS) * DIM // 8
        assert stats["compression_vs_float32"] == 32.0
        assert "python" in stats["tags"]

    def test_save_load_round_trip(self, store, embedder, tmp_path):
        path = store.save(tmp_path / "kb")
        assert path.endswith(".npz")
        loaded = BinaryVectorStore.load(path, embedder)
        assert len(loaded) == len(store)
        assert loaded.search("docker containers", k=3) == store.search("docker containers", k=3)
        assert loaded.search("framework", all_tags=["python", "web"]) == \
            store.search("framework", all_tags=["python", "web"])
        loaded.add(["elixir functional language on the beam"], tags=[["elixir"]])
        assert loaded.search("elixir beam", k=1)[0].tags == ("elixir",)

    def test_load_rejects_different_embedder(self, store, tmp_path):
        path = store.save(tmp_path / "kb.npz")
        other = FunctionEmbedder(_toy_embed, name="other-model")
        with pytest.raises(ValueError, match="embedder"):
            BinaryVectorStore.load(path, other)
        assert len(BinaryVectorStore.load(path, other, check_embedder=False)) == len(DOCS)

    def test_unserializable_metadata_raises_on_save(self, embedder, tmp_path):
        s = BinaryVectorStore(embedder, center=False)
        s.add(["doc"], metadata=[{"obj": object()}])
        with pytest.raises(TypeError, match="JSON"):
            s.save(tmp_path / "kb.npz")
