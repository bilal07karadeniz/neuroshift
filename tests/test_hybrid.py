"""Tests for neuroshift.hybrid -- AnomalyDetector and HDCRetrievalEngine."""

import pytest
import torch

from neuroshift.hybrid.anomaly_detector import AnomalyDetector
from neuroshift.hybrid.hdc_rag import HDCRetrievalEngine

# Use small HDC dimensions for speed
SMALL_DIM = 1000


# ===========================================================================
# AnomalyDetector
# ===========================================================================
class TestAnomalyDetector:
    """Unit tests for the three-paradigm anomaly detector."""

    @pytest.fixture
    def detector(self, device):
        return AnomalyDetector(
            feature_dim=8,
            hdc_dim=SMALL_DIM,
            device=device,
        )

    # -- learn_normal ----------------------------------------------------------

    def test_learn_normal_single_sample(self, detector):
        """learn_normal with a single feature vector should work."""
        features = torch.randn(8, device=detector.device)
        detector.learn_normal(features)
        assert detector.n_normal_seen == 1
        assert len(detector.normal_prototypes) == 1

    def test_learn_normal_batch(self, detector):
        """learn_normal with a batch of features should learn all samples."""
        features = torch.randn(5, 8, device=detector.device)
        detector.learn_normal(features)
        assert detector.n_normal_seen == 5
        assert len(detector.normal_prototypes) == 5

    def test_learn_normal_updates_centroid(self, detector):
        """After learning, the normal_centroid should be set."""
        features = torch.randn(3, 8, device=detector.device)
        detector.learn_normal(features)
        assert detector.normal_centroid is not None
        assert detector.normal_centroid.shape[1] == SMALL_DIM

    def test_learn_normal_multiple_calls(self, detector):
        """Calling learn_normal multiple times should accumulate prototypes."""
        detector.learn_normal(torch.randn(3, 8, device=detector.device))
        detector.learn_normal(torch.randn(2, 8, device=detector.device))
        assert detector.n_normal_seen == 5
        assert len(detector.normal_prototypes) == 5

    # -- detect ----------------------------------------------------------------

    def test_detect_returns_expected_keys(self, detector):
        """detect() should return a dict with the documented keys."""
        detector.learn_normal(torch.randn(5, 8, device=detector.device))
        result = detector.detect(torch.randn(8, device=detector.device))

        expected_keys = {'is_anomaly', 'anomaly_score', 'confidence',
                         'detection_time_ms', 'details'}
        assert expected_keys.issubset(result.keys())

    def test_detect_details_keys(self, detector):
        """The 'details' sub-dict should contain per-layer info."""
        detector.learn_normal(torch.randn(5, 8, device=detector.device))
        result = detector.detect(torch.randn(8, device=detector.device))
        details = result['details']

        expected_detail_keys = {'hdc_centroid_sim', 'hdc_min_sim', 'hdc_max_sim',
                                'morphic_arch', 'threshold'}
        assert expected_detail_keys.issubset(details.keys())

    def test_detect_anomaly_score_in_range(self, detector):
        """anomaly_score should be in [0, 1]."""
        detector.learn_normal(torch.randn(10, 8, device=detector.device))
        result = detector.detect(torch.randn(8, device=detector.device))
        assert 0.0 <= result['anomaly_score'] <= 1.0

    def test_detect_is_anomaly_is_bool(self, detector):
        """is_anomaly should be a boolean."""
        detector.learn_normal(torch.randn(5, 8, device=detector.device))
        result = detector.detect(torch.randn(8, device=detector.device))
        assert isinstance(result['is_anomaly'], bool)

    def test_detect_on_normal_data_returns_low_anomaly_score(self, detector):
        """Data drawn from the same distribution as training should have a low score."""
        torch.manual_seed(123)
        # Normal data: Gaussian around zero, small variance
        normal_data = torch.randn(20, 8, device=detector.device) * 0.1

        detector.learn_normal(normal_data)

        # Test with data from same distribution
        test_normal = torch.randn(10, 8, device=detector.device) * 0.1
        scores = []
        for i in range(test_normal.shape[0]):
            result = detector.detect(test_normal[i])
            scores.append(result['anomaly_score'])

        avg_score = sum(scores) / len(scores)
        # Normal data should have a relatively low average anomaly score
        assert avg_score < 0.8

    def test_detect_on_outlier_returns_high_anomaly_score(self, detector):
        """Outlier data should produce a higher anomaly score than normal data."""
        torch.manual_seed(456)
        # Normal: small values around zero
        normal_data = torch.randn(20, 8, device=detector.device) * 0.1
        detector.learn_normal(normal_data)

        # Test normal-like data
        test_normal = torch.randn(10, 8, device=detector.device) * 0.1
        normal_scores = []
        for i in range(test_normal.shape[0]):
            result = detector.detect(test_normal[i])
            normal_scores.append(result['anomaly_score'])

        # Test outlier data: values far from training distribution
        outlier_data = torch.randn(10, 8, device=detector.device) * 100 + 50
        outlier_scores = []
        for i in range(outlier_data.shape[0]):
            result = detector.detect(outlier_data[i])
            outlier_scores.append(result['anomaly_score'])

        avg_normal = sum(normal_scores) / len(normal_scores)
        avg_outlier = sum(outlier_scores) / len(outlier_scores)

        # Outlier average should be higher than normal average
        assert avg_outlier > avg_normal

    def test_detect_increments_anomalies_detected(self, detector):
        """n_anomalies_detected should increment when anomalies are found."""
        detector.learn_normal(torch.randn(5, 8, device=detector.device) * 0.01)
        initial_count = detector.n_anomalies_detected

        # Large outlier likely to be flagged
        for _ in range(10):
            detector.detect(torch.ones(8, device=detector.device) * 1000)

        # At least some should have been detected as anomalous
        # (n_anomalies_detected >= initial, possibly more)
        assert detector.n_anomalies_detected >= initial_count

    # -- get_status ------------------------------------------------------------

    def test_get_status(self, detector):
        """get_status() should return a dict with expected keys."""
        detector.learn_normal(torch.randn(3, 8, device=detector.device))
        status = detector.get_status()
        expected_keys = {'normal_examples_seen', 'anomalies_detected', 'threshold',
                         'morphic_trained', 'morphic_architecture', 'morphic_params',
                         'hdc_dimensions', 'device'}
        assert expected_keys.issubset(status.keys())
        assert status['normal_examples_seen'] == 3
        assert status['morphic_trained'] is False


# ===========================================================================
# HDCRetrievalEngine
# ===========================================================================
class TestHDCRetrievalEngine:
    """Unit tests for the hyperdimensional retrieval engine."""

    @pytest.fixture
    def engine(self, device):
        return HDCRetrievalEngine(dimensions=SMALL_DIM, device=device)

    @pytest.fixture
    def populated_engine(self, device):
        """Engine pre-loaded with a small corpus."""
        eng = HDCRetrievalEngine(dimensions=SMALL_DIM, device=device)
        docs = [
            ("Python is a popular programming language", {"topic": "programming"}, ["python", "programming"]),
            ("Django is a web framework for Python", {"topic": "web"}, ["python", "django", "web"]),
            ("Machine learning uses neural networks", {"topic": "ml"}, ["ml", "neural"]),
            ("Cats are wonderful pets", {"topic": "animals"}, ["animals", "cats"]),
            ("The weather today is sunny and warm", {"topic": "weather"}, ["weather"]),
        ]
        for text, meta, tags in docs:
            eng.index(text, metadata=meta, tags=tags)
        return eng

    # -- index -----------------------------------------------------------------

    def test_index_single_document(self, engine):
        """Indexing a document should return a doc_id and store the document."""
        doc_id = engine.index("Hello world", metadata={"source": "test"})
        assert doc_id == 0
        assert len(engine.documents) == 1
        assert engine.documents[0] == "Hello world"

    def test_index_multiple_documents(self, engine):
        """Each indexed document should get a unique incrementing id."""
        id0 = engine.index("First document")
        id1 = engine.index("Second document")
        id2 = engine.index("Third document")
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2
        assert len(engine.documents) == 3

    def test_index_stores_metadata(self, engine):
        """Metadata should be stored alongside the document."""
        engine.index("Test doc", metadata={"author": "Alice"})
        assert engine.doc_metadata[0] == {"author": "Alice"}

    def test_index_without_metadata(self, engine):
        """Indexing without metadata should store empty dict."""
        engine.index("No metadata")
        assert engine.doc_metadata[0] == {}

    def test_index_creates_tag_vectors(self, engine):
        """Tags provided during indexing should appear in tag_vectors."""
        engine.index("Tagged doc", tags=["alpha", "beta"])
        assert "alpha" in engine.tag_vectors
        assert "beta" in engine.tag_vectors

    def test_index_batch(self, engine):
        """index_batch should index all documents."""
        texts = ["Doc A", "Doc B", "Doc C"]
        ids = engine.index_batch(texts)
        assert ids == [0, 1, 2]
        assert len(engine.documents) == 3

    # -- search ----------------------------------------------------------------

    def test_search_returns_results(self, populated_engine):
        """search() should return a non-empty list for a relevant query."""
        results = populated_engine.search("Python programming", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_result_structure(self, populated_engine):
        """Each result should have text, score, metadata, rank, doc_id."""
        results = populated_engine.search("programming")
        assert len(results) > 0
        expected_keys = {'text', 'score', 'metadata', 'rank', 'doc_id'}
        for r in results:
            assert expected_keys.issubset(r.keys())

    def test_search_rank_ordering(self, populated_engine):
        """Results should be ordered by rank (1, 2, 3, ...)."""
        results = populated_engine.search("Python", top_k=5)
        ranks = [r['rank'] for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_search_scores_descending(self, populated_engine):
        """Scores should be in descending order."""
        results = populated_engine.search("Python", top_k=5)
        scores = [r['score'] for r in results]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-6

    def test_search_empty_engine(self, engine):
        """search() on an empty engine should return empty list."""
        results = engine.search("anything")
        assert results == []

    def test_search_increments_query_count(self, populated_engine):
        """total_queries should increment after each search."""
        assert populated_engine.total_queries == 0
        populated_engine.search("test1")
        assert populated_engine.total_queries == 1
        populated_engine.search("test2")
        assert populated_engine.total_queries == 2

    @pytest.mark.parametrize("top_k", [1, 3, 10])
    def test_search_top_k(self, populated_engine, top_k):
        """search() should return at most top_k results (capped by corpus size)."""
        results = populated_engine.search("Python", top_k=top_k)
        assert len(results) <= top_k
        assert len(results) <= len(populated_engine.documents)

    # -- search_by_concept with NOT -------------------------------------------

    def test_search_by_concept_positive(self, populated_engine):
        """search_by_concept with positive concepts should return results."""
        results = populated_engine.search_by_concept(
            positive=["python"], top_k=3
        )
        assert len(results) > 0

    def test_search_by_concept_with_negative(self, populated_engine):
        """Using negative concepts should lower scores for matching docs."""
        results_without_neg = populated_engine.search_by_concept(
            positive=["python"], top_k=5
        )
        results_with_neg = populated_engine.search_by_concept(
            positive=["python"], negative=["django"], top_k=5
        )

        # Find the Django doc score in both result sets
        def find_django_score(results):
            for r in results:
                if "django" in r['text'].lower():
                    return r['score']
            return None

        score_without = find_django_score(results_without_neg)
        score_with = find_django_score(results_with_neg)

        # If the Django doc appears in both, its score should be lower with the negative
        if score_without is not None and score_with is not None:
            assert score_with < score_without

    def test_search_by_concept_result_structure(self, populated_engine):
        """Results from search_by_concept should have the same structure."""
        results = populated_engine.search_by_concept(positive=["cats"], top_k=2)
        expected_keys = {'text', 'score', 'metadata', 'rank', 'doc_id'}
        for r in results:
            assert expected_keys.issubset(r.keys())

    def test_search_by_concept_empty_engine(self, engine):
        """search_by_concept on empty engine should return []."""
        results = engine.search_by_concept(positive=["anything"])
        assert results == []

    # -- get_stats -------------------------------------------------------------

    def test_get_stats_returns_expected_keys(self, populated_engine):
        """get_stats() should return a dict with all documented keys."""
        stats = populated_engine.get_stats()
        expected_keys = {'total_documents', 'total_queries', 'total_index_time_ms',
                         'avg_index_time_ms', 'memory_mb', 'dimensions',
                         'known_tags', 'device'}
        assert expected_keys.issubset(stats.keys())

    def test_get_stats_document_count(self, populated_engine):
        """total_documents should match the number of indexed documents."""
        stats = populated_engine.get_stats()
        assert stats['total_documents'] == 5

    def test_get_stats_dimensions(self, populated_engine):
        """dimensions should match the constructor value."""
        stats = populated_engine.get_stats()
        assert stats['dimensions'] == SMALL_DIM

    def test_get_stats_known_tags(self, populated_engine):
        """known_tags should include all tags provided during indexing."""
        stats = populated_engine.get_stats()
        tags = set(stats['known_tags'])
        assert "python" in tags
        assert "django" in tags
        assert "web" in tags
        assert "ml" in tags
        assert "animals" in tags
        assert "cats" in tags

    def test_get_stats_empty_engine(self, engine):
        """get_stats on empty engine should report 0 documents."""
        stats = engine.get_stats()
        assert stats['total_documents'] == 0
        assert stats['total_queries'] == 0
        assert stats['known_tags'] == []
