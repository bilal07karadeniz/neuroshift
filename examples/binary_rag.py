"""
Binary semantic retrieval in ~40 lines.

Real embeddings (meaning), stored as bits (32x smaller), searched with
XOR + popcount, saved to a single .npz file.

Run:  pip install "neuroshift[embed]"  &&  python examples/binary_rag.py
"""

import os
import tempfile

from neuroshift.retrieval import BinaryVectorStore, SentenceTransformerEmbedder


DOCS = [
    ("The automobile engine needs repair after the breakdown.", ["vehicles"]),
    ("Cooking pasta requires boiling salted water.", ["food"]),
    ("Django is a batteries-included Python web framework.", ["python", "web"]),
    ("Flask is a lightweight Python micro web framework.", ["python", "web"]),
    ("FastAPI builds typed Python APIs on top of Starlette.", ["python", "web"]),
    ("Express is a minimal web framework for Node.js.", ["javascript", "web"]),
    ("Pandas provides dataframes for data analysis in Python.", ["python", "data"]),
    ("Bicycles need their chains oiled regularly.", ["vehicles"]),
    ("A sourdough starter needs regular feeding with flour.", ["food"]),
    ("PostgreSQL is a relational database with strong SQL support.", ["database"]),
]


def show(title, results):
    print(f"\n{title}")
    for r in results:
        print(f"  {r.score:+.3f}  {r.text}")


def main():
    store = BinaryVectorStore(SentenceTransformerEmbedder())
    store.add([d for d, _ in DOCS], tags=[t for _, t in DOCS])

    # No shared words with the answer: works because meaning comes from the embedding model.
    show('search("my car is broken")', store.search("my car is broken", k=2))

    show('search("build a REST API", all_tags=["python"])',
         store.search("build a REST API", k=3, all_tags=["python"]))

    # AND: documents must be close to every positive concept.
    show('search_composite(["Python", "web framework"])',
         store.search_composite(["Python", "web framework"], k=3))

    # NOT demotes rather than removes; for hard exclusion use exclude_tags.
    show('search_composite(["python web framework"], negative=["Flask"])',
         store.search_composite(["python web framework"], negative=["Flask"], k=3))

    stats = store.stats()
    print(f"\n{stats['documents']} docs, {stats['bytes_per_doc']} bytes each "
          f"({stats['compression_vs_float32']:.0f}x smaller than float32)")

    with tempfile.TemporaryDirectory() as tmp:
        path = store.save(os.path.join(tmp, "kb.npz"))
        print(f"saved to one file: {os.path.getsize(path):,} bytes (codes + texts + tags)")
        reloaded = BinaryVectorStore.load(path, store.embedder)
        assert reloaded.search("my car is broken", k=1)[0].text == DOCS[0][0]


if __name__ == "__main__":
    main()
