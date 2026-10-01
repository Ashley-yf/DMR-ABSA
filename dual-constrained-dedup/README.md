# RAG Deduplication Module

Removes near-duplicate synthetic samples with a **dual constraint**:

1. **Aspect constraint** — the set of aspect terms must be identical;
2. **Semantic constraint** — sentence-embedding cosine similarity ≥ `SIMILARITY_THRESHOLD` (default 0.9).

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, short-text friendly).

## Files

| File | Role |
|---|---|
| `config.py` | All settings: input/output dirs, threshold, embedding model |
| `rag_retrieval.py` | Main entry: load → per-sample dedup → renumber IDs → write `*_RAG.json` + stats |
| `explain_aspect_matching.py` | Debug helper: explains aspect-matching decisions |
| `test_structure.py` | Lightweight sanity test of module structure |
| `rag_outputs/` | Example outputs shipped with the repo |

## Usage

```bash
# 1) point INPUT_DIR in config.py at your evaluated data (evaluation/outputs by default)
# 2) run
python dual-constrained-dedup/rag_retrieval.py
```

Results: `rag_outputs/<name>_RAG.json` with sequential local IDs, plus
processing/dedup-rate logs in `logs/`.

Useful env override: `RAG_EMBEDDING_MODEL=/path/to/local/model` (skips HF download).
