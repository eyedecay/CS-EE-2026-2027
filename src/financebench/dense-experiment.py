import json
import os
import time
from datasets import load_dataset
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from collections import OrderedDict

RESULTS_DIR = "results/financebench"
K_VALUES = [1, 2, 3]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_corpus(dataset):
    """
    Build a deduplicated corpus of evidence page texts from the dataset
    Args:
        dataset (Dataset): Hugging Face FinanceBench dataset
    Returns:
        tuple: list of unique text strings, dict mapping text to metadata
    """
    seen = OrderedDict()
    for row in dataset:
        for e in row["evidence"]:
            text = e["evidence_text_full_page"]
            if text not in seen:
                seen[text] = {"doc_name": e["doc_name"], "idx": len(seen)}
    return list(seen.keys()), {text: meta for text, meta in seen.items()}


def get_correct_indices(row, lookup):
    """
    Get indices of correct evidence chunks for a question
    Args:
        row (dict): A single FinanceBench example
        lookup (dict): Mapping from text string to metadata dict with "idx"
    Returns:
        list: Sorted unique corpus indices of correct chunks
    """
    indices = []
    for e in row["evidence"]:
        meta = lookup.get(e["evidence_text_full_page"])
        if meta is not None:
            indices.append(meta["idx"])
    return sorted(set(indices))


class TimedStreamer(BaseStreamer):
    """
    Streamer that records the time of the first generated token
    Attributes:
        first_token_time (float or None): Time when the first token was produced
    """
    def __init__(self):
        super().__init__()
        self.first_token_time = None

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


def compute_metrics(retrieved, correct, k):
    """
    Compute recall@k, precision@k, and MRR@k for a single query
    Args:
        retrieved (list): Ranked list of retrieved corpus indices
        correct (list): List of correct corpus indices
        k (int): Cutoff value
    Returns:
        tuple: (recall, precision, mrr) as floats
    """
    correct_set = set(correct)
    retrieved_set = set(retrieved[:k])
    hits = len(correct_set & retrieved_set)
    recall = hits / len(correct_set) if correct_set else 0.0
    precision = hits / k
    mrr = 0.0
    for rank, idx in enumerate(retrieved[:k], 1):
        if idx in correct_set:
            mrr = 1.0 / rank
            break
    return recall, precision, mrr


def main():
    """
    Run the dense retrieval experiment on FinanceBench using FAISS + SentenceTransformer
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_dataset("PatronusAI/financebench", split="train")

    corpus_texts, corpus_lookup = build_corpus(dataset)
    n_docs = len(set(m["doc_name"] for m in corpus_lookup.values()))
    print(f"Corpus: {len(corpus_texts)} unique chunks from {n_docs} documents")

    t_embed = time.perf_counter()
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    corpus_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"Document embedding time: {time.perf_counter() - t_embed:.2f}s")

    faiss.normalize_L2(corpus_embeddings)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    max_k = max(K_VALUES)

    for i, row in enumerate(dataset):
        query = row["question"]
        correct = get_correct_indices(row, corpus_lookup)
        question_type = row["question_type"]

        t0 = time.perf_counter()
        query_emb = embedder.encode([query], convert_to_numpy=True)
        query_embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        faiss.normalize_L2(query_emb)
        top_k = min(max_k, len(corpus_texts))
        scores, idxs = index.search(query_emb, top_k)
        search_time = time.perf_counter() - t1

        retrieval_time = query_embed_time + search_time
        retrieved_indices = idxs[0].tolist()

        entry = {
            "query_id": i,
            "question_type": question_type,
            "retrieval_time_ms": retrieval_time * 1000,
            "query_embed_time_ms": query_embed_time * 1000,
            "search_time_ms": search_time * 1000,
        }
        for k in K_VALUES:
            rec, prec, mrr = compute_metrics(retrieved_indices, correct, k)
            entry[f"recall@{k}"] = rec
            entry[f"precision@{k}"] = prec
            entry[f"mrr@{k}"] = mrr

        best_idx = retrieved_indices[0]
        context = corpus_texts[best_idx][:2000]
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        streamer = TimedStreamer()
        t_gen = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, streamer=streamer, max_new_tokens=50, do_sample=False)
        gen_time = time.perf_counter() - t_gen
        ttft = streamer.first_token_time - t_gen if streamer.first_token_time else None

        entry["ttft_ms"] = ttft * 1000 if ttft else None
        entry["gen_time_ms"] = gen_time * 1000

        results.append(entry)

    path = os.path.join(RESULTS_DIR, "dense_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
