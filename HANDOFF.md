# Handoff: deep comparison of `neuroshift.retrieval`

Status notes for continuing this work in a new (local) session. Written 2026-10-07.

## What exists

- `neuroshift/retrieval/`: binary embedding search. 1-bit sign or SimHash codes, word-major XOR+popcount scan, Hamming shortlist (k×10), then asymmetric rescoring (float query · ±1 doc bits). Also tag filters, AND/NOT concept queries, and `.npz` save/load. 50 tests in `tests/test_retrieval.py`.
- `benchmarks/bench_retrieval.py`: first benchmark (SciFact/NFCorpus × MiniLM/BGE-small, 1M-vector scale test).
- `benchmarks/embed_beir.py`: embeds BEIR datasets per model and caches them to `benchmarks/.cache/*.npy`. Models: `minilm`, `bge-small`, `embeddinggemma-2`, `mxbai-large`.
- `benchmarks/deep_compare.py`:
  - `quality`: runs every method on identical cached embeddings and reports nDCG@10, R@100, and Δ vs float32 with a paired-bootstrap 95% CI. Results go to `benchmarks/results/quality-*.json`, with per-query scores kept.
  - `scale`: 1M clustered synthetic vectors against FAISS (flat, SQ8, PQ, RaBitQ, HNSW, binary flat/HNSW) and usearch (f32/b1 HNSW). **Not run yet.**
- Cached embeddings (committed): MiniLM and BGE-small for scifact, nfcorpus and arguana.
- `benchmarks/results/quality-{scifact,nfcorpus,arguana}.json`: complete for MiniLM, BGE-small and BM25. Print the tables with `python -c "import json; from benchmarks.deep_compare import print_quality; print_quality('scifact', json.load(open('benchmarks/results/quality-scifact.json')))"`.

## Results so far (nDCG@10, % of float32 with the same model)

| method | bytes | SciFact MiniLM | SciFact BGE | NFCorpus MiniLM | NFCorpus BGE | ArguAna MiniLM | ArguAna BGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8 scalar (SQ8) | 384 | 99.9 | 100.0 | 99.9 | 99.9 | 100.0 | 100.1 |
| PQ, 192 B (in-domain trained) | 192 | 100.7 | 99.6 | 100.0 | 100.2 | 99.8 | 99.7 |
| ours: simhash 4x + rescore | 192 | 97.4 | 96.5 | 97.0 | 96.2 | 98.8 | 96.4 |
| PQ, 48 B | 48 | 97.6 | 93.0 | 97.3 | 93.1 | 92.3 | 89.7 |
| ours: sign + rescore | 48 | 96.6 | 92.5 | 93.4 | 92.7 | 94.1 | 90.8 |
| ours: sign + rescore + center | 48 | 94.9 | 91.1 | 85.3 | 90.0 | 94.3 | 91.9 |
| RaBitQ 1-bit (FAISS) | 56 | 95.3 | 95.3 | 94.3 | 94.6 | 94.3 | 95.2 |
| sign, Hamming only (= FAISS IndexBinaryFlat) | 48 | 91.2 | 87.8 | 87.3 | 81.3 | 89.1 | 82.7 |
| BM25 | – | 86.8 | 78.5 | 84.7 | 78.1 | 69.1 | 57.4 |

Conclusions so far:
- At 48 B, in-domain-trained PQ beats our sign + rescore in 4 of 6 settings; ours wins on ArguAna. RaBitQ (56 B, no rescore) beats ours in 5 of 6 and ties the sixth. At 192 B, PQ (~100%) beats our simhash (96–99%) everywhere. Int8 is lossless at 4x. Ours clearly beats only raw Hamming, MRL truncation on non-MRL models, and BM25.
- Correctness check: our Hamming-only ranking is identical to FAISS IndexBinaryFlat (same nDCG). FAISS's C++ scan is about 2.5x faster than our numpy scan.
- Centering hurt on SciFact and NFCorpus but helped slightly on ArguAna. It is off by default; the effect is dataset-dependent.
- `ms/q` values in the quality runs are noisy because jobs shared 4 CPU cores. Use only the `scale` run for speed.

## Novelty verdict (literature review, with sources)

The core method is not novel:
- The Hugging Face / mixedbread "Embedding Quantization" blog (2024-03-22, https://huggingface.co/blog/embedding-quantization) describes the same pipeline: threshold at 0, Hamming shortlist with `rescore_multiplier`, then float query × binary docs rescoring. It reports retention of 96.45% for mxbai-embed-large-v1 and 93.79% for all-MiniLM-L6-v2 on MTEB retrieval.
- Asymmetric float-query vs binary-doc distances: Gordo et al., CVPR 2011 / TPAMI 2014. Binary candidates then continuous reranking: BPR (ACL 2021). Vespa (2021). Qdrant BQ (2023). Elasticsearch BBQ (2024). pgvector 0.7 `binary_quantize` (2024).
- RaBitQ (SIGMOD 2024, arXiv 2405.12497) is a stronger, theoretically bounded version.
- A 2026 thesis (Awale, UAH) finds post-hoc PQ leads the Pareto frontier for BGE. Our data agrees.
- Concept algebra (min over positives − w·max over negatives) is a minor variant of known score combinations (fuzzy Boolean IR, Qdrant recommend API).

Possible genuinely new directions:
1. Measure how binarization or quantization affects negation and exclusion queries (NevIR, ExcluIR, QUEST). No such study was found.
2. Use per-query adaptive shortlist sizes driven by error estimates (as RaBitQ's bounds allow), instead of a fixed k×10.
3. Characterize when centering vs rotation helps for the float-query × sign-doc estimator on real text embeddings (see Xiao, arXiv 2605.17524).

## Remaining TODO

1. **Embed the big models** (needs a GPU; on the 4-core CPU they ran at 0.1–0.5 docs/s):
   `python -m benchmarks.embed_beir --models embeddinggemma-2 mxbai-large --datasets scifact nfcorpus arguana`
   - EmbeddingGemma 2 (`google/embeddinggemma-2`, released 2026-10-06, 768-d, MRL 512/256/128):
     - Needs `pillow` and `torchvision` installed, even for text only.
     - Do **not** use float16 (it returns NaNs); use bf16 or fp32.
     - Its `max_seq_length` defaults to effectively unlimited; consider `model.max_seq_length = 1024`.
     - Prompts are already set in `embed_beir.py`: query `task: search result | query: `, doc `title: {title|none} | text: {text}`.
   - mxbai-embed-large-v1 lets you compare directly with the HF blog's published 96.45%.
   - For EmbeddingGemma 2, the MRL rows are a fair test: Google claims near-lossless quality down to 256-d. Compare MRL against binary against PQ at equal bytes.
2. **Run quality on all models:** `python -m benchmarks.deep_compare quality`
3. **Run scale on a quiet machine:** `python -m benchmarks.deep_compare scale --threads 1`, then `--threads 4`.
4. **Database comparison: is a DB needed at all?**
   - pgvector: `binary_quantize` + `bit_hamming_ops` HNSW + float rerank, and plain HNSW.
   - Qdrant: `docker run -p 6333:6333 qdrant/qdrant` with binary quantization + oversampling + rescore.
   - Measure quality on the same embeddings, latency, memory and ingest time.
   - Docker Hub rate-limited the cloud session, so this was not done.
5. **End-to-end RAG example:** retrieve with `BinaryVectorStore`, then generate an answer with the Claude API (needs `ANTHROPIC_API_KEY`).
6. **Write the final report** and update the README with honest positioning.
   - The library is a clean, dependency-light, single-file implementation of a known technique.
   - It is not state of the art at equal bytes.
   - Consider adding PQ, int8 + rescore, or RaBitQ backends.
7. Known bug outside this work: `hybrid/anomaly_detector.py` encoding is scale-invariant, so `test_detect_on_outlier_returns_high_anomaly_score` fails.
