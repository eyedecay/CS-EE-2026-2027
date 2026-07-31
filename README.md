# Computer Science Extended Essay 2026-2027

A comparison of dense vs sparse retrieval in RAG using the FinanceBench dataset.

- **Dense**: FAISS + `all-MiniLM-L6-v2` embeddings
- **Sparse**: BM25 (bm25s)
- **LLM for generation**: Qwen2.5-0.5B
- **Metrics**: recall@k, precision@k, MRR@k (k=1,2,3), retrieval time, TTFT, generation time
