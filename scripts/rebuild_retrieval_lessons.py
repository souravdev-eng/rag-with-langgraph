#!/usr/bin/env python3
"""Rebuild dense/sparse, reranking, and MMR retrieval lessons."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_dense_sparse() -> None:
    build_standard_lesson(
        "09-hybrid-search-strategies/1-densesparse.ipynb",
        {
            "stage": "Retrieval and reranking",
            "title": "Sparse, Dense, and Hybrid Retrieval",
            "difficulty": "Intermediate",
            "key_idea": "Sparse and dense branches fail differently. Evaluate each branch before fusing ranks, and never add hybrid complexity without a measured reason.",
            "summary": "This notebook compares BM25 with a low-rank dense LSA representation on the shared eight-document corpus and seven answerable golden questions. Reciprocal-rank fusion combines the branches without pretending their raw scores share a scale.",
            "why": "Sparse retrieval rewards exact terms; dense representations can group correlated vocabulary but introduce model and compression error. Directly averaging incomparable scores can make a hybrid system harder to reason about.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| BM25, low-rank dense retrieval, hit rate/MRR, reciprocal-rank fusion | Neural embedding download, ANN indexes, large-scale latency benchmark |",
            "mental_model": "```text\nquery -> BM25 rank ----\\\n                         reciprocal-rank fusion -> final rank\nquery -> dense LSA rank-/\n```",
            "setup_code": r'''
import numpy as np
from rank_bm25 import BM25Okapi
from rag_101 import TfidfVectorizer, load_corpus, load_golden_questions
from rag_101.lexical import tokenize

documents = load_corpus()
questions = [item for item in load_golden_questions() if item.relevant_doc_ids]
document_tokens = [tokenize(document.content) for document in documents]
len(documents), len(questions)
''',
            "how_it_works": "BM25 uses term frequency, document frequency, and length normalization. The dense branch projects TF-IDF vectors into a five-dimensional latent space with SVD; unlike a neural embedding it cannot understand unseen synonyms, but it clearly demonstrates dense compression. Reciprocal-rank fusion (RRF) uses positions rather than incompatible score magnitudes.",
            "baseline_text": "BM25 is the sparse baseline. It builds an inverted-term-style scorer over tokenized documents and returns exact-term-sensitive rankings.",
            "baseline_code": r'''
bm25 = BM25Okapi(document_tokens)

def sparse_rank(query: str) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(documents)), key=lambda index: (-scores[index], documents[index].id))
    return [documents[index].id for index in order]

[(question.id, sparse_rank(question.question)[:2]) for question in questions]
''',
            "technique_text": "The dense branch fits TF-IDF once, performs truncated SVD, and compares projected query/document vectors by dot product. Five dimensions deliberately create a lossy representation so the experiment exposes rather than hides the branch trade-off.",
            "technique_code": r'''
vectorizer = TfidfVectorizer().fit(document.content for document in documents)
document_matrix = np.array(vectorizer.transform(document.content for document in documents))
_, _, right_vectors = np.linalg.svd(document_matrix, full_matrices=False)
latent_basis = right_vectors[:5].T
dense_documents = document_matrix @ latent_basis

def dense_rank(query: str) -> list[str]:
    query_vector = np.array(vectorizer.transform_one(query)) @ latent_basis
    scores = dense_documents @ query_vector
    order = sorted(range(len(documents)), key=lambda index: (-scores[index], documents[index].id))
    return [documents[index].id for index in order]

[(question.id, dense_rank(question.question)[:2]) for question in questions]
''',
            "experiment_text": "We hold corpus, questions, cutoff, and labels fixed. RRF assigns each document `1 / (60 + rank)` from each branch. Hit rate@1 and MRR are computed from the first relevant document.",
            "experiment_code": r'''
def rrf_rank(query: str, constant: int = 60, weights: tuple[float, float] = (1.0, 1.0)) -> list[str]:
    branch_ranks = (sparse_rank(query), dense_rank(query))
    scores = {document.id: 0.0 for document in documents}
    for weight, ranking in zip(weights, branch_ranks, strict=True):
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += weight / (constant + rank)
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))

def evaluate(rank_function) -> dict[str, float]:
    hits, reciprocal_ranks = [], []
    for question in questions:
        ranking = rank_function(question.question)
        relevant = set(question.relevant_doc_ids)
        first = next((rank for rank, doc_id in enumerate(ranking, 1) if doc_id in relevant), None)
        hits.append(ranking[0] in relevant)
        reciprocal_ranks.append(1 / first if first else 0.0)
    return {"hit_rate@1": sum(hits) / len(hits), "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)}

results = {
    "bm25": evaluate(sparse_rank),
    "dense_lsa_5d": evaluate(dense_rank),
    "equal_rrf": evaluate(rrf_rank),
    "sparse_weighted_rrf": evaluate(lambda query: rrf_rank(query, weights=(2.0, 1.0))),
}
results
''',
            "evaluation": "On this exact-term-heavy set, BM25 reaches **1.00 hit rate@1** while the deliberately compressed LSA branch reaches **0.857**. Equal-weight RRF also falls to **0.857** because the weak branch can overturn one correct top result. Weighting the validated sparse branch 2:1 restores **1.00**. Hybrid is not automatically safer; its weights need evidence.",
            "checks_code": r'''
assert results["bm25"]["hit_rate@1"] == 1.0
assert results["dense_lsa_5d"]["hit_rate@1"] == 6 / 7
assert results["equal_rrf"]["hit_rate@1"] == 6 / 7
assert results["sparse_weighted_rrf"]["hit_rate@1"] == 1.0
assert all(sorted(rank_function(question.question)) == sorted(document.id for document in documents)
           for rank_function in (sparse_rank, dense_rank, rrf_rank) for question in questions)
print("Sparse/dense/hybrid checks passed.")
''',
            "decision_guide": "| Query shape | Start with |\n|---|---|\n| IDs, error codes, exact names | Sparse/BM25 |\n| Paraphrases and conceptual language | Evaluated neural dense model |\n| Mixed traffic | Hybrid after branch-level measurement |\n| Incomparable branch scores | Rank fusion or calibrated normalization |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Hybrid worse than best branch | Weak branch adds noise | Measure branches and tune/disable by segment |\n| One branch dominates | Raw scores averaged | Use rank fusion or calibrated scores |\n| Dense misses exact code | Representation smooths rare token | Keep sparse branch |\n| Offline score is perfect | Golden set favors exact terms | Add paraphrase and hard-negative cases |",
            "production_notes": "### Observability\nLog per-branch ranks/scores, overlap, fusion contribution, query segment, latency, and index/model version.\n\n### Safety and Guardrails\nApply the same authorization filter to every branch before fusion.\n\n### Latency and Cost\nParallel branches reduce wall time but increase total work; enforce per-branch timeouts and a deterministic fallback.",
            "practice": "Add two paraphrased questions with no distinctive source terms and compare the three methods before changing any weights.",
            "recall": "Toggle - Recall: Why not average BM25 and cosine scores directly?\nTheir ranges and meanings differ.\n\nToggle - Recall: What did hybrid add here?\nRobustness to the weaker dense branch, not a quality improvement over BM25.",
            "sources": "- [BM25 paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)\n- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)\n- Repository golden dataset",
            "confidence": "High for the bounded shared-corpus experiment",
            "next_review": "Add neural embeddings and paraphrase-heavy labels",
        },
    )


def build_reranking() -> None:
    build_standard_lesson(
        "09-hybrid-search-strategies/2-reranking.ipynb",
        {
            "stage": "Retrieval and reranking",
            "title": "Reranking: Retrieve Broad, Score Narrow",
            "difficulty": "Intermediate",
            "key_idea": "A reranker can only reorder candidates it receives. Candidate recall and reranker precision must be measured separately.",
            "summary": "This notebook builds a transparent two-stage example where term frequency promotes a keyword-stuffed distractor. A bounded pairwise proxy penalizes repetition and restores the relevant MFA recovery document—only when candidate depth is at least two.",
            "why": "Fast first-stage retrieval is optimized for recall and can return shallow matches. More expensive query-document scoring can improve precision, but it cannot recover a missing document and adds latency.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Candidate depth, transparent reranking features, recall@k, MRR, latency boundary | Downloaded cross-encoder, LLM judge, production benchmark |",
            "mental_model": "```text\nquery -> fast retriever -> top-N candidates -> pairwise reranker -> top-k context\n               recall gate ^                    precision gate\n```",
            "setup_code": r'''
from collections import Counter
import re

query = "reset MFA after losing phone"
candidates = [
    {"id": "stuffed", "relevant": False, "text": "Reset reset MFA MFA lost phone keywords for a marketing taxonomy."},
    {"id": "recovery", "relevant": True, "text": "If your phone is lost, contact support to reset multi-factor authentication (MFA) after identity verification."},
    {"id": "password", "relevant": False, "text": "Reset a forgotten password from the account sign-in page."},
    {"id": "devices", "relevant": False, "text": "Register a new phone after signing in on an existing trusted device."},
]

def tokens(text: str) -> list[str]: return re.findall(r"[a-z0-9]+", text.lower())
''',
            "how_it_works": "Stage one cheaply scores query-term frequency. Stage two sees each query-document pair, rewards unique query coverage and an ordered lost-phone-to-reset relation, and penalizes repeated query terms. The proxy exposes mechanics; a trained cross-encoder learns richer interactions.",
            "baseline_text": "The first-stage scorer sums occurrences of query terms. Keyword stuffing therefore outranks the genuinely useful recovery instruction.",
            "baseline_code": r'''
query_terms = tokens(query)
def first_stage_score(document: dict) -> int:
    counts = Counter(tokens(document["text"]))
    return sum(counts[term] for term in query_terms)

first_stage = sorted(candidates, key=lambda doc: (-first_stage_score(doc), doc["id"]))
[(doc["id"], first_stage_score(doc)) for doc in first_stage]
''',
            "technique_text": "The reranker uses only inspectable features: unique coverage, a recovery-instruction signal, and repetition penalty. It is deliberately not presented as a substitute for a trained relevance model.",
            "technique_code": r'''
def rerank_score(document: dict) -> float:
    document_tokens = tokens(document["text"])
    counts = Counter(document_tokens)
    unique_coverage = len(set(query_terms) & set(document_tokens)) / len(set(query_terms))
    recovery_signal = float("phone" in document_tokens and "reset" in document_tokens and "verification" in document_tokens)
    repetition_penalty = sum(max(0, counts[term] - 1) for term in set(query_terms)) / len(set(query_terms))
    return unique_coverage + 0.75 * recovery_signal - 0.5 * repetition_penalty

def rerank(candidate_set: list[dict]) -> list[dict]:
    return sorted(candidate_set, key=lambda doc: (-rerank_score(doc), doc["id"]))

[(doc["id"], round(rerank_score(doc), 3)) for doc in rerank(first_stage)]
''',
            "experiment_text": "We vary first-stage candidate depth from one to four. Recall@N asks whether the relevant document reaches stage two; reciprocal rank measures its final position after reranking.",
            "experiment_code": r'''
experiment_rows = []
for candidate_depth in range(1, len(candidates) + 1):
    selected = first_stage[:candidate_depth]
    reranked = rerank(selected)
    first_relevant_rank = next((rank for rank, doc in enumerate(reranked, 1) if doc["relevant"]), None)
    experiment_rows.append({
        "candidate_depth": candidate_depth,
        "candidate_recall": float(any(doc["relevant"] for doc in selected)),
        "top_document": reranked[0]["id"],
        "reciprocal_rank": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    })
experiment_rows
''',
            "evaluation": "At depth **1**, candidate recall is zero and reranking cannot help. At depth **2 or greater**, the recovery document is present and moves to rank 1 (MRR 1.0). The example proves the two-stage dependency, not real cross-encoder accuracy.",
            "checks_code": r'''
assert first_stage[0]["id"] == "stuffed"
assert experiment_rows[0]["candidate_recall"] == 0.0
assert all(row["top_document"] == "recovery" and row["reciprocal_rank"] == 1.0 for row in experiment_rows[1:])
assert rerank_score(next(doc for doc in candidates if doc["id"] == "recovery")) > rerank_score(first_stage[0])
print("Reranking checks passed.")
''',
            "decision_guide": "| Situation | Choice |\n|---|---|\n| Candidate recall already low | Improve first-stage retrieval first |\n| Good recall, weak top-1 precision | Add evaluated reranker |\n| Tight latency budget | Smaller candidate depth/model or no reranker |\n| High-stakes ranking | Calibrated model plus human-reviewed hard negatives |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Reranker never finds answer | Relevant item absent | Raise/tune candidate recall |\n| Latency spikes | Candidate depth/model too large | Batch, cap depth, timeout, fallback |\n| Offline gains vanish | Easy or leaked labels | Add temporal/hard-negative evaluation |\n| Pairwise scores misread as probabilities | Uncalibrated logits | Treat as ranks unless calibrated |",
            "production_notes": "### Observability\nTrack candidate recall, before/after rank, score deltas, depth, model/version, batch latency, and fallback rate.\n\n### Safety and Guardrails\nNever rerank unauthorized candidates; filtering must precede both stages.\n\n### Latency and Cost\nMeasure p50/p95 by candidate depth and include timeout behavior in quality evaluation.",
            "practice": "Add a relevant document that ranks fifth in stage one. Plot final MRR against candidate depth and choose the smallest acceptable depth.",
            "recall": "Toggle - Recall: What is the reranker's hard limit?\nIt cannot select a document absent from the candidate set.\n\nToggle - Recall: Why evaluate candidate recall separately?\nIt distinguishes first-stage misses from ordering errors.",
            "sources": "- [BERT passage reranking paper](https://arxiv.org/abs/1901.04085)\n- [Sentence Transformers cross-encoder reranking](https://www.sbert.net/examples/cross_encoder/applications/README.html)",
            "confidence": "High for the two-stage dependency demonstration",
            "next_review": "Benchmark a trained reranker on shared hard negatives",
        },
    )


def build_mmr() -> None:
    build_standard_lesson(
        "09-hybrid-search-strategies/3-mmr.ipynb",
        {
            "stage": "Retrieval and reranking",
            "title": "Maximal Marginal Relevance: Balance Relevance and Diversity",
            "difficulty": "Intermediate",
            "key_idea": "MMR selects a set, not just high-scoring items. The diversity weight can reduce redundancy—or admit irrelevant novelty when pushed too far.",
            "summary": "This notebook implements MMR from scratch over five incident-response candidates. It compares pure relevance (`lambda=1.0`), a moderate diversity setting (`0.75`), and an aggressive setting (`0.5`).",
            "why": "Top-k retrieval often returns near-duplicates that consume context without adding evidence. Diversity can broaden coverage, but novelty is not the same as relevance.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Greedy MMR objective, lambda sweep, redundancy and relevance checks | Learned diversification, submodular guarantees, large-vector index integration |",
            "mental_model": "```text\nnext = argmax(lambda * relevance - (1-lambda) * max_similarity_to_selected)\n```",
            "setup_code": r'''
from math import sqrt

candidates = {
    "incident_basics": {"relevance": 0.95, "vector": [1.0, 0.0]},
    "incident_checklist": {"relevance": 0.92, "vector": [0.98, 0.02]},
    "postmortem": {"relevance": 0.82, "vector": [0.70, 0.70]},
    "customer_comms": {"relevance": 0.78, "vector": [0.20, 0.98]},
    "unrelated_novelty": {"relevance": 0.30, "vector": [-1.0, 0.0]},
}

def cosine(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    )
''',
            "how_it_works": "MMR starts with the strongest relevance candidate. Each later step subtracts the maximum similarity to anything already selected. `lambda=1` ignores redundancy; smaller values reward novelty more strongly.",
            "baseline_text": "Pure relevance takes the three highest relevance scores. The first two are almost identical in vector direction, so much of the context budget is redundant.",
            "baseline_code": r'''
relevance_top3 = sorted(candidates, key=lambda name: (-candidates[name]["relevance"], name))[:3]
relevance_top3
''',
            "technique_text": "The implementation exposes the per-step relevance and redundancy terms. Tie-breaking is deterministic by candidate name.",
            "technique_code": r'''
def mmr_select(lambda_mult: float, k: int = 3) -> list[str]:
    if not 0 <= lambda_mult <= 1: raise ValueError("lambda_mult must be between 0 and 1.")
    selected = []
    while len(selected) < k:
        def score(name: str) -> float:
            relevance = candidates[name]["relevance"]
            redundancy = max(
                (cosine(candidates[name]["vector"], candidates[item]["vector"]) for item in selected),
                default=0.0,
            )
            return lambda_mult * relevance - (1 - lambda_mult) * redundancy
        remaining = [name for name in candidates if name not in selected]
        selected.append(max(remaining, key=lambda name: (score(name), name)))
    return selected

{value: mmr_select(value) for value in (1.0, 0.75, 0.5)}
''',
            "experiment_text": "For each lambda we record mean relevance, maximum pairwise similarity (redundancy), and whether the deliberately irrelevant novelty item enters the set.",
            "experiment_code": r'''
def set_metrics(selection: list[str]) -> dict[str, float | bool]:
    similarities = [
        cosine(candidates[left]["vector"], candidates[right]["vector"])
        for index, left in enumerate(selection) for right in selection[index + 1:]
    ]
    return {
        "mean_relevance": sum(candidates[name]["relevance"] for name in selection) / len(selection),
        "max_pair_similarity": max(similarities),
        "mean_pair_similarity": sum(similarities) / len(similarities),
        "contains_irrelevant_novelty": "unrelated_novelty" in selection,
    }

experiment_results = {value: {"selection": mmr_select(value), **set_metrics(mmr_select(value))} for value in (1.0, 0.75, 0.5)}
experiment_results
''',
            "evaluation": "`lambda=1.0` keeps maximum relevance but selects the near-duplicate checklist. `0.75` adds customer communication coverage and lowers mean pair similarity, although the near-duplicate pair still remains. At `0.5`, the unrelated opposite-direction item enters—proof that aggressive diversity can harm relevance.",
            "checks_code": r'''
assert experiment_results[1.0]["selection"] == relevance_top3
assert experiment_results[0.75]["mean_pair_similarity"] < experiment_results[1.0]["mean_pair_similarity"]
assert not experiment_results[0.75]["contains_irrelevant_novelty"]
assert experiment_results[0.5]["contains_irrelevant_novelty"]
assert experiment_results[0.75]["mean_relevance"] > experiment_results[0.5]["mean_relevance"]
print("MMR checks passed.")
''',
            "decision_guide": "| Need | Lambda direction |\n|---|---|\n| Highest precision / single fact | Toward 1.0 |\n| Multi-aspect answer with duplicates | Moderate diversity |\n| Broad exploration | Lower only with relevance floor |\n| Independent evidence | Diversify by source/metadata as well as vector |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Irrelevant result enters | Diversity overweighted | Raise lambda/add relevance floor |\n| Duplicates remain | Lambda too high or similarity poor | Lower lambda/evaluate representation |\n| Different sources still repeat claim | Vector diversity ≠ source independence | Add source/parent constraints |\n| Results unstable | Ties/order undocumented | Deterministic tie-break and version config |",
            "production_notes": "### Observability\nLog lambda, fetch depth, selected ranks, relevance, redundancy penalty, source diversity, and final context use.\n\n### Safety and Guardrails\nDiversity never overrides authorization or minimum relevance.\n\n### Latency and Cost\nMMR is greedy over the fetched pool; bound `fetch_k` and avoid recomputing pairwise similarities.",
            "practice": "Add a second customer-communication document and determine whether vector, source, or category diversity best matches the answer goal.",
            "recall": "Toggle - Recall: What does lambda control?\nThe relevance-versus-redundancy trade-off.\n\nToggle - Recall: Why can low lambda fail?\nA novel but irrelevant candidate can beat a relevant similar one.",
            "sources": "- [Carbonell and Goldstein: MMR](https://dl.acm.org/doi/10.1145/290941.291025)\n- [LangChain MMR API reference](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html)",
            "confidence": "High for the transparent MMR objective",
            "next_review": "Evaluate lambda with real duplicate clusters and source diversity",
        },
    )


if __name__ == "__main__":
    build_dense_sparse()
    build_reranking()
    build_mmr()
