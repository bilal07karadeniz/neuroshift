"""Tests for neuroshift.hdc.engine -- HyperdimensionalEngine, OneShotClassifier, AnalogyEngine."""

import pytest
import torch

from neuroshift.hdc.engine import (
    HyperdimensionalEngine,
    OneShotClassifier,
    AnalogyEngine,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
SMALL_DIM = 1000  # keep tests fast


# ===========================================================================
# HyperdimensionalEngine
# ===========================================================================
class TestHyperdimensionalEngine:
    """Unit tests for the core HDC engine."""

    @pytest.fixture
    def engine(self, device):
        return HyperdimensionalEngine(dimensions=SMALL_DIM, device=device)

    # -- random_bipolar -------------------------------------------------------

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_random_bipolar_shape(self, engine, n):
        """random_bipolar(n) should return shape (n, dim)."""
        hv = engine.random_bipolar(n)
        assert hv.shape == (n, SMALL_DIM)

    def test_random_bipolar_values(self, engine):
        """Every element must be exactly -1 or +1."""
        hv = engine.random_bipolar(20)
        unique = torch.unique(hv)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    # -- bind -----------------------------------------------------------------

    def test_bind_self_inverse(self, engine):
        """bind(bind(a, b), a) should recover b (self-inverse property)."""
        a = engine.random_bipolar(1)
        b = engine.random_bipolar(1)
        recovered = engine.bind(engine.bind(a, b), a)
        # For bipolar {-1,+1}, element-wise multiply is exactly self-inverse.
        assert torch.allclose(recovered, b)

    def test_bind_shape_preserved(self, engine):
        """bind should preserve the input shape."""
        a = engine.random_bipolar(3)
        b = engine.random_bipolar(3)
        result = engine.bind(a, b)
        assert result.shape == a.shape

    # -- bundle ---------------------------------------------------------------

    def test_bundle_majority_vote(self, engine):
        """Bundle of three identical vectors should return the same vector."""
        v = engine.random_bipolar(1)
        bundled = engine.bundle(v.expand(3, -1).clone())
        assert torch.allclose(bundled, v)

    def test_bundle_output_bipolar(self, engine):
        """Bundle output must be bipolar {-1, +1}."""
        vectors = engine.random_bipolar(7)
        bundled = engine.bundle(vectors)
        unique = torch.unique(bundled)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    # -- permute --------------------------------------------------------------

    def test_permute_creates_different_vector(self, engine):
        """A permuted vector should differ from the original."""
        v = engine.random_bipolar(1)
        p = engine.permute(v, shifts=1)
        # Extremely unlikely to be the same for dim=1000
        assert not torch.allclose(v, p)

    def test_permute_preserves_shape(self, engine):
        """Permute should not change the tensor shape."""
        v = engine.random_bipolar(1)
        p = engine.permute(v, shifts=3)
        assert p.shape == v.shape

    def test_permute_preserves_values(self, engine):
        """Permute is a circular shift -- the multiset of values is unchanged."""
        v = engine.random_bipolar(1)
        p = engine.permute(v, shifts=7)
        assert torch.sort(v.flatten()).values.equal(torch.sort(p.flatten()).values)

    # -- similarity -----------------------------------------------------------

    def test_similarity_same_vector(self, engine):
        """Cosine similarity of a vector with itself should be 1.0."""
        v = engine.random_bipolar(1)
        sim = engine.similarity(v, v)
        assert pytest.approx(sim.item(), abs=1e-5) == 1.0

    def test_similarity_orthogonal_near_zero(self, engine):
        """Two independent random bipolar vectors should have near-zero similarity."""
        a = engine.random_bipolar(1)
        b = engine.random_bipolar(1)
        sim = engine.similarity(a, b).item()
        # With dim=1000 the expected std is ~1/sqrt(1000) ~ 0.032
        assert abs(sim) < 0.2  # generous bound

    # -- encode_text ----------------------------------------------------------

    def test_encode_text_returns_correct_shape(self, engine):
        """encode_text should return shape (1, dim)."""
        hv = engine.encode_text("hello world")
        assert hv.shape == (1, SMALL_DIM)

    def test_encode_text_bipolar(self, engine):
        """encode_text output must be bipolar."""
        hv = engine.encode_text("testing one two three")
        unique = torch.unique(hv)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    def test_encode_text_same_input_high_similarity(self, engine):
        """Encoding the same text twice should produce highly similar hypervectors.

        Because bundle breaks ties randomly, exact equality is not guaranteed.
        However the deterministic component (cached codebook) dominates, so
        cosine similarity should be well above zero.
        """
        hv1 = engine.encode_text("deterministic test")
        hv2 = engine.encode_text("deterministic test")
        sim = engine.similarity(hv1, hv2).item()
        # With even-count bundles ~31% of dims get random tie-breaks,
        # so similarity is roughly 0.69.  We use a conservative lower bound.
        assert sim > 0.3

    @pytest.mark.parametrize("text", ["", "   "])
    def test_encode_text_empty_string(self, engine, text):
        """Empty or whitespace-only text should still return valid (1, dim) output."""
        hv = engine.encode_text(text)
        assert hv.shape == (1, SMALL_DIM)


# ===========================================================================
# OneShotClassifier
# ===========================================================================
class TestOneShotClassifier:
    """Unit tests for instant one-shot classification."""

    @pytest.fixture
    def engine(self, device):
        return HyperdimensionalEngine(dimensions=SMALL_DIM, device=device)

    @pytest.fixture
    def classifier(self, engine):
        return OneShotClassifier(engine)

    def test_learn_and_predict_correct_class(self, engine, classifier):
        """After learning one example per class, predict should return the correct label."""
        hv_a = engine.random_bipolar(1)
        hv_b = engine.random_bipolar(1)
        classifier.learn("A", hv_a)
        classifier.learn("B", hv_b)

        label, score, scores = classifier.predict(hv_a)
        assert label == "A"
        assert score > 0

    def test_predict_returns_tuple(self, engine, classifier):
        """predict should return (label: str, confidence: float, scores: dict)."""
        classifier.learn("X", engine.random_bipolar(1))
        result = classifier.predict(engine.random_bipolar(1))
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)
        assert isinstance(result[2], dict)

    def test_predict_unknown_when_empty(self, engine, classifier):
        """Predict on an untrained classifier should return 'unknown'."""
        label, score, scores = classifier.predict(engine.random_bipolar(1))
        assert label == "unknown"
        assert score == 0.0
        assert scores == {}

    def test_accuracy_improves_with_more_examples(self, engine, classifier):
        """Feeding more examples of each class should maintain or improve accuracy."""
        torch.manual_seed(42)

        # Create two well-separated prototypes
        proto_a = engine.random_bipolar(1)
        proto_b = engine.random_bipolar(1)

        # Generate test data: noisy copies of the prototypes
        test_data = []
        for _ in range(20):
            noisy_a = proto_a.clone()
            # Flip ~10% of bits
            flip_mask = torch.rand_like(noisy_a) < 0.1
            noisy_a[flip_mask] *= -1
            test_data.append(("A", noisy_a))

            noisy_b = proto_b.clone()
            flip_mask = torch.rand_like(noisy_b) < 0.1
            noisy_b[flip_mask] *= -1
            test_data.append(("B", noisy_b))

        # Learn with 1 example each
        classifier.learn("A", proto_a)
        classifier.learn("B", proto_b)
        acc_1 = classifier.accuracy_on(test_data)

        # Learn with additional examples (noisy)
        for _ in range(5):
            noisy_a = proto_a.clone()
            flip_mask = torch.rand_like(noisy_a) < 0.05
            noisy_a[flip_mask] *= -1
            classifier.learn("A", noisy_a)

            noisy_b = proto_b.clone()
            flip_mask = torch.rand_like(noisy_b) < 0.05
            noisy_b[flip_mask] *= -1
            classifier.learn("B", noisy_b)

        acc_more = classifier.accuracy_on(test_data)

        # With more training, accuracy should be at least as good
        assert acc_more >= acc_1 - 0.05  # small tolerance for stochasticity

    @pytest.mark.parametrize("n_classes", [2, 5, 10])
    def test_multiclass_classification(self, engine, n_classes):
        """Classifier should handle multiple classes."""
        clf = OneShotClassifier(engine)
        hvs = {}
        for i in range(n_classes):
            hv = engine.random_bipolar(1)
            clf.learn(f"class_{i}", hv)
            hvs[f"class_{i}"] = hv

        # Each prototype should predict its own class
        for label, hv in hvs.items():
            predicted, _, _ = clf.predict(hv)
            assert predicted == label


# ===========================================================================
# AnalogyEngine
# ===========================================================================
class TestAnalogyEngine:
    """Unit tests for algebraic analogy reasoning."""

    @pytest.fixture
    def engine(self, device):
        return HyperdimensionalEngine(dimensions=SMALL_DIM, device=device)

    @pytest.fixture
    def analogy(self, engine):
        return AnalogyEngine(engine)

    def test_define_creates_concept(self, analogy):
        """define() should register the concept."""
        analogy.define("apple")
        assert "apple" in analogy.concepts

    def test_define_with_custom_hv(self, engine, analogy):
        """define() with a provided HV should store that exact vector."""
        hv = engine.random_bipolar(1)
        analogy.define("custom", hv=hv)
        assert torch.allclose(analogy.concepts["custom"], hv)

    def test_define_composite(self, engine, analogy):
        """define_composite should create a concept from role-filler bindings."""
        analogy.define("gender")
        analogy.define("male")
        analogy.define("role")
        analogy.define("royalty")
        analogy.define_composite("king", {"gender": "male", "role": "royalty"})
        assert "king" in analogy.concepts
        assert analogy.concepts["king"].shape == (1, SMALL_DIM)

    def test_solve_analogy_returns_expected_answer(self, engine, analogy):
        """Structured analogy: king:queen :: man:? should return 'woman' as top result."""
        # Define role/filler atoms
        analogy.define("gender")
        analogy.define("male")
        analogy.define("female")
        analogy.define("role")
        analogy.define("royalty")
        analogy.define("commoner")

        # Define composites
        analogy.define_composite("king", {"gender": "male", "role": "royalty"})
        analogy.define_composite("queen", {"gender": "female", "role": "royalty"})
        analogy.define_composite("man", {"gender": "male", "role": "commoner"})
        analogy.define_composite("woman", {"gender": "female", "role": "commoner"})

        results = analogy.solve_analogy("king", "queen", "man",
                                        candidates=["woman", "royalty", "commoner", "male", "female"])
        # "woman" should rank highest
        top_name = results[0][0]
        assert top_name == "woman"

    def test_solve_analogy_returns_list_of_tuples(self, analogy):
        """solve_analogy should return List[Tuple[str, float]]."""
        analogy.define("a")
        analogy.define("b")
        analogy.define("c")
        analogy.define("d")
        results = analogy.solve_analogy("a", "b", "c")
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_query_role_works(self, engine, analogy):
        """query_role('king', 'gender') should return 'male' near the top."""
        analogy.define("gender")
        analogy.define("male")
        analogy.define("female")
        analogy.define("role")
        analogy.define("royalty")
        analogy.define_composite("king", {"gender": "male", "role": "royalty"})

        results = analogy.query_role("king", "gender")
        assert isinstance(results, list)
        assert len(results) > 0
        # "male" should be the highest-ranked filler
        top_name = results[0][0]
        assert top_name == "male"

    def test_query_role_excludes_composite_and_role(self, analogy):
        """query_role should not include the composite or the role itself in results."""
        analogy.define("gender")
        analogy.define("male")
        analogy.define_composite("king", {"gender": "male"})

        results = analogy.query_role("king", "gender")
        result_names = [r[0] for r in results]
        assert "king" not in result_names
        assert "gender" not in result_names
