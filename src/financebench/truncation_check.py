import csv
import os
import statistics

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATASET = "financebench"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_DIR = "results/financebench"
TRIALS_SUBDIR = os.path.join(RESULTS_DIR, "trials")
OUT_CSV = os.path.join(TRIALS_SUBDIR, "evidence_positions.csv")
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]


def build_corpus(dataset):
    """
    Reconstruct the experiment corpus
    Args:
        dataset (Dataset): The FinanceBench dataset split
    Returns:
        tuple: (corpus list of page texts, lookup dict mapping page text to index)
    """
    corpus = []
    lookup = {}
    for row in dataset:
        for evidence in row["evidence"]:
            text = evidence["evidence_text_full_page"]
            if text not in lookup:
                lookup[text] = len(corpus)
                corpus.append(text)
    return corpus, lookup


def locate_evidence(tokenizer, dataset):
    """
    find the token span each gold evidence string occupies inside its full page
    Args:
        tokenizer (PreTrainedTokenizer): MiniLM tokenizer 
        dataset (Dataset): The FinanceBench dataset
    Returns:
        list: One dict per gold evidence span with its query, question type, and
            token positions for the page start, span start, span end, and page end
    """
    records = []
    for query_index, row in enumerate(dataset):
        for evidence_index, evidence in enumerate(row["evidence"]):
            page = evidence["evidence_text_full_page"]
            span = evidence["evidence_text"]
            start_char = page.find(span)
            if start_char == -1:
                continue
            records.append({
                "query_index": query_index,
                "evidence_index": evidence_index,
                "question_type": row["question_type"],
                "evidence_page_num": evidence["evidence_page_num"],
                "doc_name": evidence["doc_name"],
                "page_tokens": len(tokenizer(page)["input_ids"]),
                "evidence_start_token": len(tokenizer(page[:start_char])["input_ids"]),
                "evidence_end_token": len(tokenizer(page[:start_char + len(span)])["input_ids"]),
                "evidence_chars": len(span),
            })
    return records


def summarise(records, limit):
    """
    Report how many gold evidence spans fall wholly inside the sequence limit
    Args:
        records (list): Span records from locate_evidence
        limit (int): Model maximum sequence length 
    Returns:
        dict: Share of spans starting, ending, and lying wholly within the limit
    """
    starts = [r["evidence_start_token"] for r in records]
    ends = [r["evidence_end_token"] for r in records]
    within = sum(1 for s, e in zip(starts, ends) if s <= limit and e <= limit)
    return {
        "n_spans": len(records),
        "starts_within": sum(1 for s in starts if s <= limit),
        "ends_within": sum(1 for e in ends if e <= limit),
        "fully_within": within,
        "entirely_beyond": sum(1 for s in starts if s >= limit),
        "median_start": statistics.median(starts),
        "median_end": statistics.median(ends),
        "mean_end": statistics.mean(ends),
    }


def main():

    os.makedirs(TRIALS_SUBDIR, exist_ok=True)
    dataset = load_dataset("PatronusAI/financebench", split="train")
    model = SentenceTransformer(EMBEDDING_MODEL)
    limit = model.max_seq_length

    corpus, _ = build_corpus(dataset)
    records = locate_evidence(model.tokenizer, dataset)

    fieldnames = [
        "query_index", "evidence_index", "question_type", "doc_name", "evidence_page_num",
        "page_tokens", "evidence_start_token", "evidence_end_token", "evidence_chars",
        "page_truncated", "evidence_start_within_limit", "evidence_fully_within_limit",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["page_truncated"] = record["page_tokens"] > limit
            row["evidence_start_within_limit"] = record["evidence_start_token"] <= limit
            row["evidence_fully_within_limit"] = (
                record["evidence_start_token"] <= limit and record["evidence_end_token"] <= limit
            )
            writer.writerow(row)

    truncated_pages = sum(1 for record in records if record["page_tokens"] > limit)
    total = summarise(records, limit)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Maximum sequence length: {limit} tokens")
    print(f"Corpus pages: {len(corpus)}")
    print(f"Gold evidence spans located: {total['n_spans']}\n")

    share = lambda count: 100 * count / total["n_spans"]
    print("Where gold evidence sits inside its page:")
    print(f"  starts within first {limit} tokens    {total['starts_within']:4d}/{total['n_spans']}"
          f"  ({share(total['starts_within']):5.1f}%)")
    print(f"  ends within first {limit} tokens      {total['ends_within']:4d}/{total['n_spans']}"
          f"  ({share(total['ends_within']):5.1f}%)")
    print(f"  fully within first {limit} tokens     {total['fully_within']:4d}/{total['n_spans']}"
          f"  ({share(total['fully_within']):5.1f}%)")
    print(f"  entirely beyond the window           {total['entirely_beyond']:4d}/{total['n_spans']}"
          f"  ({share(total['entirely_beyond']):5.1f}%)")
    print(f"  start token: median {total['median_start']:.0f}, mean {statistics.mean(r['evidence_start_token'] for r in records):.0f}")
    print(f"  end token:   median {total['median_end']:.0f}, mean {total['mean_end']:.0f}")

    print(f"\nPages exceeding the limit: {truncated_pages}/{total['n_spans']} gold spans "
          f"({100 * truncated_pages / total['n_spans']:.1f}%)")

    print("\nBy question type (first gold evidence span per query):")
    first_by_query = {}
    for record in records:
        first_by_query.setdefault(record["query_index"], record)
    for qtype in QUESTION_TYPES:
        subset = [r for r in first_by_query.values() if r["question_type"] == qtype]
        if not subset:
            continue
        within = sum(1 for r in subset
                     if r["evidence_start_token"] <= limit and r["evidence_end_token"] <= limit)
        starts = statistics.median(r["evidence_start_token"] for r in subset)
        ends = statistics.median(r["evidence_end_token"] for r in subset)
        print(f"  {qtype:20} start median {starts:4.0f}  end median {ends:4.0f}  "
              f"fully within limit {within:3d}/{len(subset)}")

    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()