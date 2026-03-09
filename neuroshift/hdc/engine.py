"""
HYPERDIMENSIONAL COMPUTING ENGINE
==================================
A fundamentally different computing paradigm. Instead of training neural networks
with gradient descent over thousands of iterations, HDC encodes information into
high-dimensional random vectors (10,000 dimensions) where:

- Learning is INSTANT (one-shot, no training loop)
- Reasoning is ALGEBRAIC (analogy = vector arithmetic)
- Memory is DISTRIBUTED (robust to noise and hardware failure)
- Computation is PARALLEL (massive GPU speedup)

Based on Kanerva's theory of hyperdimensional computing.

Use cases:
  - Few-shot classification (learn from 1-5 examples)
  - Analogy reasoning (king:queen :: man:?)
  - Real-time anomaly detection
  - Edge AI (no training infrastructure needed)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time


class HyperdimensionalEngine:
    """Core GPU-accelerated hyperdimensional computing engine."""

    def __init__(self, dimensions: int = 10000, device: str = "auto"):
        self.dim = dimensions
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._char_codebook: Dict[str, torch.Tensor] = {}
        self._pos_codebook: Dict[int, torch.Tensor] = {}
        self._level_cache: Optional[List[torch.Tensor]] = None

    def random_bipolar(self, n: int = 1) -> torch.Tensor:
        """Generate random bipolar hypervectors in {-1, +1}^d."""
        return torch.randint(0, 2, (n, self.dim), device=self.device, dtype=torch.float32) * 2 - 1

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binding: element-wise multiply. Creates associations.
        KEY PROPERTY: self-inverse, so bind(bind(a,b), a) ≈ b"""
        return a * b

    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        """Bundling: element-wise majority vote. Creates superpositions.
        The result is similar to ALL inputs simultaneously."""
        summed = vectors.sum(dim=0, keepdim=True)
        ties = (summed == 0)
        if ties.any():
            summed[ties] = self.random_bipolar(1).expand_as(summed)[ties]
        return summed.sign()

    def permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """Permutation: circular shift. Encodes order/position.
        Each shift creates a quasi-orthogonal vector."""
        return torch.roll(v, shifts=shifts, dims=-1)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between hypervectors."""
        return F.cosine_similarity(a.view(-1, self.dim), b.view(-1, self.dim), dim=-1)

    def _get_char_hv(self, char: str) -> torch.Tensor:
        if char not in self._char_codebook:
            self._char_codebook[char] = self.random_bipolar(1)
        return self._char_codebook[char]

    def _get_pos_hv(self, pos: int) -> torch.Tensor:
        if pos not in self._pos_codebook:
            self._pos_codebook[pos] = self.random_bipolar(1)
        return self._pos_codebook[pos]

    def _get_word_hv(self, word: str) -> torch.Tensor:
        """Get or create a hypervector for a word (cached)."""
        if not hasattr(self, '_word_codebook'):
            self._word_codebook: Dict[str, torch.Tensor] = {}
        if word not in self._word_codebook:
            self._word_codebook[word] = self.random_bipolar(1)
        return self._word_codebook[word]

    def encode_text(self, text: str, ngram_size: int = 3) -> torch.Tensor:
        """Encode text using word-level + character n-gram hybrid encoding.

        Words capture semantic tokens, n-grams capture morphological similarity.
        Both are bundled together for a rich representation.
        """
        text = text.lower().strip()

        all_hvs = []

        # Word-level encoding: each word gets a position-bound hypervector
        words = text.split()
        for i, word in enumerate(words):
            # Strip basic punctuation
            word = word.strip('.,!?;:()[]"\'')
            if not word:
                continue
            word_hv = self._get_word_hv(word)
            pos_hv = self._get_pos_hv(i)
            all_hvs.append(self.bind(word_hv, pos_hv))

        # Also add unpositioned word HVs (bag-of-words component)
        # This makes classification more robust to word order variation
        for word in words:
            word = word.strip('.,!?;:()[]"\'')
            if not word:
                continue
            all_hvs.append(self._get_word_hv(word))

        if not all_hvs:
            return self.random_bipolar(1)

        return self.bundle(torch.cat(all_hvs, dim=0))

    def encode_numeric(self, value: float, levels: int = 100) -> torch.Tensor:
        """Encode a scalar using thermometer-style level encoding.
        Nearby values get similar vectors, far values get dissimilar ones."""
        if self._level_cache is None or len(self._level_cache) != levels:
            self._level_cache = [self.random_bipolar(1)]
            flip_per_level = self.dim // levels
            for i in range(1, levels):
                prev = self._level_cache[-1].clone()
                start = ((i - 1) * flip_per_level) % self.dim
                end = min(start + flip_per_level, self.dim)
                prev[0, start:end] *= -1
                self._level_cache.append(prev)

        idx = int(max(0, min(levels - 1, value * (levels - 1))))
        return self._level_cache[idx]


class OneShotClassifier:
    """Classifier that learns from SINGLE examples. No epochs, no batches, no loss functions.

    This is the key paradigm shift: show it one example of "spam" and one of "ham",
    and it immediately classifies new text. Add more examples to improve accuracy.
    """

    def __init__(self, engine: HyperdimensionalEngine):
        self.engine = engine
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def learn(self, label: str, hypervector: torch.Tensor):
        """Learn from a single example. O(d) time, no backprop."""
        if label not in self.prototypes:
            self.prototypes[label] = hypervector.clone()
            self.counts[label] = 1
        else:
            self.prototypes[label] = self.engine.bundle(
                torch.cat([self.prototypes[label], hypervector], dim=0)
            )
            self.counts[label] += 1

    def predict(self, hypervector: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """Predict class label with confidence scores."""
        if not self.prototypes:
            return "unknown", 0.0, {}

        labels = list(self.prototypes.keys())
        protos = torch.cat([self.prototypes[l] for l in labels], dim=0)
        sims = self.engine.similarity(hypervector.expand(len(labels), -1), protos)

        scores = {labels[i]: sims[i].item() for i in range(len(labels))}
        best_idx = sims.argmax().item()
        return labels[best_idx], sims[best_idx].item(), scores

    def accuracy_on(self, data: List[Tuple[str, torch.Tensor]]) -> float:
        """Test accuracy on labeled data."""
        if not data:
            return 0.0
        correct = sum(1 for label, hv in data if self.predict(hv)[0] == label)
        return correct / len(data)


class AnalogyEngine:
    """Algebraic reasoning through hypervector arithmetic.

    The key insight: bind(A, B) captures the RELATIONSHIP between A and B.
    To solve A:B :: C:?, compute ? = bind(bind(A, B), C) and find the nearest concept.

    This works because bind is self-inverse:
      bind(bind(king, queen), man) ≈ woman
    when king and man share the "male" component, and queen and woman share "female".
    """

    def __init__(self, engine: HyperdimensionalEngine):
        self.engine = engine
        self.concepts: Dict[str, torch.Tensor] = {}

    def define(self, name: str, hv: Optional[torch.Tensor] = None):
        """Define a concept. Auto-generates random HV if none provided."""
        self.concepts[name] = hv if hv is not None else self.engine.random_bipolar(1)

    def define_composite(self, name: str, properties: Dict[str, str]):
        """Define concept as a structured bundle of role:filler bindings.

        Example: define_composite("king", {"gender": "male", "role": "royalty"})
        This creates: king = bundle(bind(gender, male) + bind(role, royalty))
        """
        components = []
        for role, filler in properties.items():
            if role not in self.concepts:
                self.define(role)
            if filler not in self.concepts:
                self.define(filler)
            components.append(self.engine.bind(self.concepts[role], self.concepts[filler]))

        self.concepts[name] = self.engine.bundle(torch.cat(components, dim=0))

    def solve_analogy(self, a: str, b: str, c: str,
                      candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Solve: A is to B as C is to ?

        Uses vector arithmetic: ? = bind(bind(A, B), C)
        Then finds nearest concept in memory.
        """
        relationship = self.engine.bind(self.concepts[a], self.concepts[b])
        predicted = self.engine.bind(relationship, self.concepts[c])

        search = candidates or list(self.concepts.keys())
        search = [s for s in search if s not in {a, b, c}]

        results = []
        for name in search:
            sim = self.engine.similarity(predicted, self.concepts[name]).item()
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query_role(self, composite_name: str, role: str) -> List[Tuple[str, float]]:
        """Unbind a role from a composite to recover its filler.

        Example: query_role("king", "gender") should return "male" as top result.
        """
        composite = self.concepts[composite_name]
        role_hv = self.concepts[role]
        unbound = self.engine.bind(composite, role_hv)

        results = []
        for name, hv in self.concepts.items():
            if name in {composite_name, role}:
                continue
            sim = self.engine.similarity(unbound, hv).item()
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
