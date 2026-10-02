import json
import os
import statistics
import time
from datasets import load_dataset
import bm25s
import Stemmer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from collections import OrderedDict

RESULTS_DIR = "results/financebench"
K_VALUES = [1, 2, 3]
LLM_NAME = "Qwen/Qwen2.5-0.5B"
AUTO_DEVICE = "cpu"
DEVICE = os.environ.get("FINBENCH_DEVICE") or AUTO_DEVICE
RUN_TAG = os.environ.get("FINBENCH_TAG", "")
IS_WARMUP = os.environ.get("FINBENCH_WARMUP") == "1"
METHOD = "sparse"
TRIALS_SUBDIR = os.path.join(RESULTS_DIR, "trials")
OUT_DIR = os.path.join(TRIALS_SUBDIR, METHOD) if RUN_TAG else RESULTS_DIR
OUT_PATH = os.path.join(OUT_DIR, f"{METHOD}_results{RUN_TAG}.json")
ACCURACY_METRICS = [f"{m}@{k}" for k in K_VALUES for m in ("recall", "precision", "mrr")]
TIMING_METRICS = [
    "query_tokenize_time_ms",
    "score_time_ms",
    "retrieval_time_ms",
    "score_to_first_token_ms",
    "prompt_prep_time_ms",
    "full_ttft_ms",
    "full_gen_time_ms",
]


def sync_device():
    """
    Block until queued device work has completed so timings include GPU execution
    """
    if DEVICE == "mps":
        torch.mps.synchronize()


def build_corpus(dataset):
    """
    Build a the document dataset
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
    Skips the initial put of the full prompt so the timing reflects prefill
    Attributes:
        first_token_time (float or None): Time when the first token was produced
    """
    def __init__(self):
        super().__init__()
        self.first_token_time = None
        self.prompt_done = False

    def put(self, value):
        if not self.prompt_done:
            self.prompt_done = True
            return
        if self.first_token_time is None:
            sync_device()
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


def save_results(results):
    """
    Save the per-query results for this trial, plus a file holding this trial's average metrics
    Args:
        results (list): Per-query result dicts collected during this trial
    """
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_PATH}")

    averages = {}
    for metric in ACCURACY_METRICS + TIMING_METRICS:
        values = [r[metric] for r in results if r.get(metric) is not None]
        averages[metric] = round(statistics.mean(values), 4) if values else None

    avg_path = os.path.join(OUT_DIR, f"{METHOD}{RUN_TAG}_avg.json")
    payload = {
        "method": METHOD,
        "trial": RUN_TAG.lstrip("_") or "single",
        "device": DEVICE,
        "n_queries": len(results),
        "mean_metrics": averages,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(avg_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {avg_path}")


def main():
    """
    Run the sparse retrieval experiment on FinanceBench using BM25 (bm25s)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_dataset("PatronusAI/financebench", split="train")

    corpus_texts, corpus_lookup = build_corpus(dataset)
    n_docs = len(set(m["doc_name"] for m in corpus_lookup.values()))
    print(f"Corpus: {len(corpus_texts)} unique chunks from {n_docs} documents")

    stemmer = Stemmer.Stemmer("english")

    t_index = time.perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    print(f"Indexing time: {time.perf_counter() - t_index:.2f}s")

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

        t_query_start = time.perf_counter()
        query_tokens = bm25s.tokenize([query], stopwords="english", stemmer=stemmer, show_progress=False)
        t_tokenize_done = time.perf_counter()
        query_tokenize_time = t_tokenize_done - t_query_start

        top_k = min(max_k, len(corpus_texts))
        result = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
        t_score_done = time.perf_counter()

        retrieval_time = t_score_done - t_query_start
        score_time = t_score_done - t_tokenize_done
        retrieved_ids = result.documents[0].tolist()

        entry = {
            "query_id": i,
            "question_type": question_type,
            "retrieval_time_ms": retrieval_time * 1000,
            "query_tokenize_time_ms": query_tokenize_time * 1000,
            "score_time_ms": score_time * 1000,
        }
        for k in K_VALUES:
            rec, prec, mrr = compute_metrics(retrieved_ids, correct, k)
            entry[f"recall@{k}"] = rec
            entry[f"precision@{k}"] = prec
            entry[f"mrr@{k}"] = mrr

        best_idx = retrieved_ids[0]
        context = corpus_texts[best_idx][:2000]
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        streamer = TimedStreamer()
        t_prompt = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_prep_time = time.perf_counter() - t_prompt

        t_gen = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, streamer=streamer, max_new_tokens=50, do_sample=False)
        sync_device()
        gen_time = time.perf_counter() - t_gen

        entry["prompt_prep_time_ms"] = prompt_prep_time * 1000
        entry["full_gen_time_ms"] = gen_time * 1000
        if streamer.first_token_time is None:
            entry["score_to_first_token_ms"] = None
            entry["full_ttft_ms"] = None
        else:
            entry["score_to_first_token_ms"] = (streamer.first_token_time - t_score_done) * 1000
            entry["full_ttft_ms"] = (streamer.first_token_time - t_query_start) * 1000

        results.append(entry)

    if IS_WARMUP:
        print("Warmup run complete, results discarded")
        return

    save_results(results)


if __name__ == "__main__":
    main()
