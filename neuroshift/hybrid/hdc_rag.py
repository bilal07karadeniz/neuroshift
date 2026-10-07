"""
HDC-RAG: HYPERDIMENSIONAL RETRIEVAL ENGINE (LEXICAL)
=====================================================
Encodes documents as bundles of random per-word hypervectors. Because word
vectors are random, this matches shared WORDS, not meanings: "car" and
"automobile" are unrelated. It is a fuzzy bag-of-words matcher.

On BEIR SciFact it reaches nDCG@10 0.25, below BM25 (0.56) and far below
embedding search (0.65), while storing 40 KB per document as float32. See
benchmarks/bench_retrieval.py.

For semantic search use neuroshift.retrieval.BinaryVectorStore, which stores
real embeddings as bits (48 bytes per document for a 384-dim model).

Kept for its compositional hypervector operations and as an HDC example.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import time
import math

from neuroshift.hdc.engine import HyperdimensionalEngine


class HDCRetrievalEngine:
    """Lexical hyperdimensional retrieval.

    Documents are encoded as bipolar {-1, +1} hypervectors (stored as
    float32) built from random word vectors, supporting algebraic query
    composition (AND, NOT on concepts). See the module docstring for its
    limits; prefer neuroshift.retrieval.BinaryVectorStore for semantic search.
    """

    def __init__(self, dimensions: int = 10000, device: str = "auto"):
        self.engine = HyperdimensionalEngine(dimensions=dimensions, device=device)
        self.dim = dimensions

        # Document store
        self.documents: List[str] = []
        self.doc_vectors: List[torch.Tensor] = []
        self.doc_metadata: List[Dict] = []

        # Tag/concept index for compositional queries
        self.tag_vectors: Dict[str, torch.Tensor] = {}

        # Stats
        self.total_queries = 0
        self.total_index_time = 0.0

    def index(self, text: str, metadata: Optional[Dict] = None, tags: Optional[List[str]] = None):
        """Index a document. Instant - no training needed.

        Args:
            text: Document text to index
            metadata: Optional metadata dict (title, source, etc.)
            tags: Optional tags for compositional queries
        """
        t0 = time.perf_counter()

        hv = self.engine.encode_text(text)
        doc_id = len(self.documents)

        self.documents.append(text)
        self.doc_vectors.append(hv)
        self.doc_metadata.append(metadata or {})

        # Index tags
        if tags:
            for tag in tags:
                tag = tag.lower()
                if tag not in self.tag_vectors:
                    self.tag_vectors[tag] = hv.clone()
                else:
                    # Bundle with existing tag vector
                    self.tag_vectors[tag] = self.engine.bundle(
                        torch.cat([self.tag_vectors[tag], hv], dim=0)
                    )

        self.total_index_time += time.perf_counter() - t0
        return doc_id

    def index_batch(self, texts: List[str], metadata_list: Optional[List[Dict]] = None,
                    tags_list: Optional[List[List[str]]] = None):
        """Index multiple documents at once."""
        ids = []
        for i, text in enumerate(texts):
            meta = metadata_list[i] if metadata_list else None
            tags = tags_list[i] if tags_list else None
            ids.append(self.index(text, meta, tags))
        return ids

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents similar to query text.

        Returns list of {text, score, metadata, rank}.
        """
        if not self.doc_vectors:
            return []

        t0 = time.perf_counter()
        self.total_queries += 1

        query_hv = self.engine.encode_text(query)

        # Compute similarities to all documents
        doc_matrix = torch.cat(self.doc_vectors, dim=0)
        query_expanded = query_hv.expand(doc_matrix.shape[0], -1)
        sims = F.cosine_similarity(query_expanded, doc_matrix, dim=-1)

        # Get top-k
        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(sims, k)

        elapsed = time.perf_counter() - t0

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            results.append({
                'text': self.documents[idx],
                'score': score,
                'metadata': self.doc_metadata[idx],
                'rank': rank + 1,
                'doc_id': idx,
            })

        return results

    def search_by_concept(self, positive: List[str], negative: Optional[List[str]] = None,
                          top_k: int = 5) -> List[Dict]:
        """Compositional search: find docs matching concept algebra.

        Example: search_by_concept(["python", "web"], negative=["django"])
        This finds docs about Python web development but NOT Django.

        Uses multi-query scoring:
          - AND: minimum similarity across all positive concepts
          - NOT: subtract similarity to negative concepts
        """
        if not self.doc_vectors:
            return []

        doc_matrix = torch.cat(self.doc_vectors, dim=0)

        # Compute similarity to each positive concept separately
        pos_sim_lists = []
        for concept in positive:
            concept = concept.lower()
            if concept in self.tag_vectors:
                hv = self.tag_vectors[concept]
            else:
                hv = self.engine.encode_text(concept)
            sims = F.cosine_similarity(hv.expand(doc_matrix.shape[0], -1), doc_matrix, dim=-1)
            pos_sim_lists.append(sims)

        if not pos_sim_lists:
            return []

        # AND semantics: use minimum similarity (doc must match ALL concepts)
        combined = torch.stack(pos_sim_lists, dim=0)
        scores = combined.min(dim=0).values

        # NOT semantics: subtract similarity to negative concepts
        if negative:
            for concept in negative:
                concept = concept.lower()
                if concept in self.tag_vectors:
                    hv = self.tag_vectors[concept]
                else:
                    hv = self.engine.encode_text(concept)
                neg_sims = F.cosine_similarity(hv.expand(doc_matrix.shape[0], -1), doc_matrix, dim=-1)
                scores = scores - 0.5 * neg_sims

        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k)

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            results.append({
                'text': self.documents[idx],
                'score': score,
                'metadata': self.doc_metadata[idx],
                'rank': rank + 1,
                'doc_id': idx,
            })

        return results

    def search_by_analogy(self, a: str, b: str, c: str, top_k: int = 5) -> List[Dict]:
        """Analogy-based search: A is to B as C is to ?

        Finds documents that have the same relationship to C
        that B has to A.

        Example: search_by_analogy("python", "django", "javascript")
        -> finds docs about JavaScript web frameworks
        """
        if not self.doc_vectors:
            return []

        hv_a = self.engine.encode_text(a)
        hv_b = self.engine.encode_text(b)
        hv_c = self.engine.encode_text(c)

        # Relationship from A to B
        relationship = self.engine.bind(hv_a, hv_b)
        # Apply to C
        query_hv = self.engine.bind(relationship, hv_c)

        doc_matrix = torch.cat(self.doc_vectors, dim=0)
        sims = F.cosine_similarity(query_hv.expand(doc_matrix.shape[0], -1), doc_matrix, dim=-1)

        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(sims, k)

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            results.append({
                'text': self.documents[idx],
                'score': score,
                'metadata': self.doc_metadata[idx],
                'rank': rank + 1,
                'doc_id': idx,
            })

        return results

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        memory_bytes = len(self.doc_vectors) * self.dim * 4  # float32
        return {
            'total_documents': len(self.documents),
            'total_queries': self.total_queries,
            'total_index_time_ms': self.total_index_time * 1000,
            'avg_index_time_ms': (self.total_index_time / max(1, len(self.documents))) * 1000,
            'memory_mb': memory_bytes / (1024 * 1024),
            'dimensions': self.dim,
            'known_tags': list(self.tag_vectors.keys()),
            'device': str(self.engine.device),
        }
