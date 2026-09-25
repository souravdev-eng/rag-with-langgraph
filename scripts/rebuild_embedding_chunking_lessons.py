#!/usr/bin/env python3
"""Rebuild embedding and semantic-chunking lessons with offline defaults."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_embedding_fundamentals() -> None:
    build_standard_lesson(
        "06-Vector-embedding-and-vector-databases/embedding.ipynb",
        {
            "stage": "Embeddings and indexes",
            "title": "Embedding Fundamentals: Geometry Before Providers",
            "difficulty": "Intermediate",
            "key_idea": "An embedding is useful only with a compatible similarity function, normalization policy, and task-specific evaluation.",
            "summary": "This notebook builds small transparent vectors, compares dot product, cosine similarity, and Euclidean distance, and shows why vector magnitude can change rankings when vectors are not normalized.",
            "why": "Vector databases expose distance settings that look interchangeable. A mismatch between model output, normalization, and index metric can silently change retrieval order.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Vector dimensions, dot/cosine/L2, normalization, ranking checks | Neural training, ANN internals, hosted-model benchmarking |",
            "mental_model": "```text\ntext -> embedding model/version -> vector -> optional normalization -> similarity metric -> ranking\n```",
            "setup_code": r'''
from math import sqrt

def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))

def norm(vector: list[float]) -> float:
    return sqrt(dot(vector, vector))

def normalize(vector: list[float]) -> list[float]:
    length = norm(vector)
    if length == 0: raise ValueError("Cannot normalize a zero vector.")
    return [value / length for value in vector]

def cosine(left: list[float], right: list[float]) -> float:
    return dot(left, right) / (norm(left) * norm(right))

def l2(left: list[float], right: list[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
''',
            "how_it_works": "Dot product rewards alignment and magnitude. Cosine divides out magnitude and compares direction. Euclidean distance measures straight-line separation. On L2-normalized vectors, cosine and Euclidean rankings are monotonically related; without normalization they can disagree.",
            "baseline_text": "The baseline ranks raw vectors by dot product without documenting magnitude. A long vector aligned with the query can dominate a closer-direction vector.",
            "baseline_code": r'''
query = [1.0, 0.0]
candidates = {
    "same_direction_large": [10.0, 0.0],
    "close_direction": [0.9, 0.1],
    "orthogonal_large": [0.0, 20.0],
}
raw_dot_ranking = sorted(candidates, key=lambda name: dot(query, candidates[name]), reverse=True)
[(name, dot(query, candidates[name])) for name in raw_dot_ranking]
''',
            "technique_text": "We compute all three measures explicitly and repeat dot-product ranking after normalization. Every compared vector has the same dimension; zero vectors are rejected rather than converted into misleading similarities.",
            "technique_code": r'''
comparison = {
    name: {
        "dot": round(dot(query, vector), 4),
        "cosine": round(cosine(query, vector), 4),
        "l2": round(l2(query, vector), 4),
        "magnitude": round(norm(vector), 4),
    }
    for name, vector in candidates.items()
}
comparison
''',
            "experiment_text": "The controlled change is normalization. We compare raw dot ranking with normalized dot, cosine, and ascending L2 rankings on the same vectors.",
            "experiment_code": r'''
normalized_query = normalize(query)
normalized_candidates = {name: normalize(vector) for name, vector in candidates.items()}
rankings = {
    "raw_dot": raw_dot_ranking,
    "normalized_dot": sorted(candidates, key=lambda name: dot(normalized_query, normalized_candidates[name]), reverse=True),
    "cosine": sorted(candidates, key=lambda name: cosine(query, candidates[name]), reverse=True),
    "normalized_l2": sorted(candidates, key=lambda name: l2(normalized_query, normalized_candidates[name])),
}
rankings
''',
            "evaluation": "Raw dot product puts the large same-direction vector first because magnitude is part of the score. After normalization, dot, cosine, and L2 agree on the full order. This geometric check does not establish semantic quality; that requires labeled queries and documents.",
            "checks_code": r'''
assert rankings["normalized_dot"] == rankings["cosine"] == rankings["normalized_l2"]
assert abs(cosine([1, 0], [10, 0]) - 1.0) < 1e-12
assert abs(l2(normalize([1, 0]), normalize([0, 1])) - sqrt(2)) < 1e-12
try:
    cosine([1.0], [1.0, 2.0])
except ValueError:
    pass
else:
    raise AssertionError("Dimension mismatch must fail.")
print("Embedding geometry checks passed.")
''',
            "decision_guide": "| Situation | Metric/policy |\n|---|---|\n| Model recommends cosine | Normalize or use cosine explicitly |\n| Magnitude carries meaning | Dot product, documented and tested |\n| Euclidean-trained representation | L2 with compatible normalization |\n| Unknown provider behavior | Inspect docs and verify rankings locally |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Ranking changes after migration | Metric/normalization changed | Log model, dimension, metric, normalization |\n| All scores near zero | Model/query domain mismatch | Evaluate labeled domain queries |\n| Insert/query dimension error | Mixed models or versions | Version indexes and reject mismatches |\n| Perfect demo scores | Tiny/easy labels | Expand and segment the golden set |",
            "production_notes": "### Observability\nRecord model/version, dimension, normalization, metric, score distribution, and zero-vector rate.\n\n### Safety and Guardrails\nEmbeddings can leak information; apply authorization before retrieval and protect vectors like derived sensitive data.\n\n### Latency and Cost\nBatch documents, cache by content hash and model version, and measure query and indexing paths separately.",
            "practice": "Add a candidate whose direction is slightly worse but magnitude is 100× larger. Predict each ranking before running it.",
            "recall": "Toggle - Recall: When do cosine and L2 rankings agree?\nFor the same L2-normalized vectors.\n\nToggle - Recall: Does a high cosine score prove relevance?\nNo; relevance depends on the model, domain, labels, and task.",
            "sources": "- [Sentence Transformers: similarity metrics](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)\n- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)",
            "confidence": "High for the demonstrated geometry",
            "next_review": "Add ANN recall and domain-mismatch experiments",
        },
    )


def build_provider_lesson() -> None:
    build_standard_lesson(
        "06-Vector-embedding-and-vector-databases/openaiembeddings.ipynb",
        {
            "stage": "Embeddings and indexes",
            "title": "Embedding Providers: Stable Contracts and Offline Evaluation",
            "difficulty": "Intermediate",
            "key_idea": "Provider swaps are index migrations. Keep model configuration explicit and compare quality, dimension, latency, and cost on one labeled dataset.",
            "summary": "This notebook defines a provider-neutral embedding contract and compares two deterministic local implementations on the shared corpus. A hosted provider can replace either implementation without changing the evaluation harness.",
            "why": "Model names, dimensions, pricing, and APIs change. Binding notebook logic directly to one hosted client makes clean runs expensive and obscures whether a migration preserved retrieval behavior.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Provider contract, model versioning, deterministic local comparison, batching | Live paid calls, current price claims, production ANN index |",
            "mental_model": "```text\ntexts -> EmbeddingProvider(model, version) -> vectors(dim) -> versioned index\nqueries -> same provider/version -----------^\n```",
            "setup_code": r'''
from hashlib import sha256
from pathlib import Path
from rag_101 import TfidfVectorizer, load_corpus, load_golden_questions

documents = load_corpus()
questions = [item for item in load_golden_questions() if item.relevant_doc_ids]
len(documents), len(questions)
''',
            "how_it_works": "A provider must embed documents and queries into one compatible space and declare its identity and dimension. The index must record that identity; mixing versions requires re-embedding or separate indexes.",
            "baseline_text": "The baseline is the fitted TF-IDF representation from the foundation lesson. It is deterministic, corpus-specific, and strong on exact terms but weak on paraphrases.",
            "baseline_code": r'''
class TfidfProvider:
    name = "local-tfidf-v1"
    def __init__(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer().fit(texts)
        self.dimension = len(self.vectorizer.vocabulary)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts)
    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.transform_one(text)

tfidf_provider = TfidfProvider([doc.content for doc in documents])
tfidf_provider.name, tfidf_provider.dimension
''',
            "technique_text": "The comparison provider uses deterministic signed feature hashing. It has fixed dimension and no fitting step, which makes updates simple but introduces collisions. It is an interface example, not a semantic neural embedding.",
            "technique_code": r'''
import math, re

class HashingProvider:
    name = "local-hashing-v1"
    def __init__(self, dimension: int = 256): self.dimension = dimension
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]
    def embed_query(self, text: str) -> list[float]: return self._embed(text)

hashing_provider = HashingProvider()
hashing_provider.name, hashing_provider.dimension
''',
            "experiment_text": "Both providers embed the same corpus and seven answerable golden questions. We use the same cosine scorer and report hit rate@1; provider identity is the only changed component.",
            "experiment_code": r'''
def dot(left, right): return sum(a * b for a, b in zip(left, right, strict=True))

def hit_rate(provider) -> float:
    vectors = provider.embed_documents([doc.content for doc in documents])
    hits = 0
    for question in questions:
        query_vector = provider.embed_query(question.question)
        best = max(zip(documents, vectors, strict=True), key=lambda pair: dot(query_vector, pair[1]))[0]
        hits += best.id in question.relevant_doc_ids
    return hits / len(questions)

results = {
    tfidf_provider.name: {"dimension": tfidf_provider.dimension, "hit_rate@1": hit_rate(tfidf_provider)},
    hashing_provider.name: {"dimension": hashing_provider.dimension, "hit_rate@1": hit_rate(hashing_provider)},
}
results
''',
            "evaluation": "TF-IDF reaches **1.00 hit rate@1** on this exact-term-heavy golden set. Hashing is deterministic and fixed-width but may lose quality through collisions. Neither is a substitute for evaluating a true semantic model on paraphrases.",
            "checks_code": r'''
assert results["local-tfidf-v1"]["hit_rate@1"] == 1.0
assert all(len(vector) == hashing_provider.dimension for vector in hashing_provider.embed_documents(["a", "b"]))
assert hashing_provider.embed_query("repeatable") == hashing_provider.embed_query("repeatable")
assert tfidf_provider.name != hashing_provider.name
print("Provider-contract checks passed.")
''',
            "decision_guide": "| Need | Choice |\n|---|---|\n| Transparent lexical baseline | TF-IDF |\n| Fixed local feature space | Hashing baseline |\n| Semantic paraphrase retrieval | Evaluated local/hosted neural model |\n| Provider migration | New index version plus parity/quality gate |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Query returns nonsense | Query/index model mismatch | Store and verify provider identity |\n| Quality drops after upgrade | Model behavior changed | Run golden regression before cutover |\n| Cost spike | Re-embedded unchanged content | Cache by content hash + model version |\n| Clean run needs credentials | No offline default | Keep deterministic local path and tag paid cells |",
            "production_notes": "### Observability\nLog provider/model/version, dimensions, batch size, latency, token/input volume, failures, and index version.\n\n### Safety and Guardrails\nSending text to a provider is data transmission; classify/redact content and use approved regions and retention terms.\n\n### Latency and Cost\nBatch indexing, cache outputs, rate-limit retries, and obtain current pricing from the provider before budgeting.",
            "practice": "Implement a third provider behind the same methods and rerun the golden set without changing evaluation code.",
            "recall": "Toggle - Recall: Why is a provider swap an index migration?\nOld and new vectors may have different dimensions and geometry.\n\nToggle - Recall: What stays fixed in a fair comparison?\nCorpus, questions, scorer, cutoff, and labels.",
            "sources": "- [LangChain embeddings interface](https://python.langchain.com/docs/concepts/embedding_models/)\n- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)\n- Repository golden dataset",
            "confidence": "High for the offline contract and regression harness",
            "next_review": "Run approved semantic providers with current cost/latency data",
        },
    )


def build_semantic_chunking() -> None:
    build_standard_lesson(
        "08-advanced-chunking-and-preprocessing/semanti_chunking.ipynb",
        {
            "stage": "Chunking",
            "title": "Semantic Chunking: Break Where Topics Change",
            "difficulty": "Intermediate",
            "key_idea": "Semantic chunking is a breakpoint policy, not magic. Compare it with a simple baseline on answer-support coverage and context noise.",
            "summary": "This notebook splits a five-sentence fixture by paragraph baseline and by adjacent-sentence TF-IDF similarity. The semantic policy detects the topic change from LangChain to Paris while preserving every sentence exactly once.",
            "why": "Fixed boundaries can join unrelated topics or cut a coherent explanation. Semantic breakpoints can help, but thresholds add model, latency, and stability trade-offs.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Sentence boundaries, adjacent similarity, threshold selection, coverage checks | Neural semantic models, multilingual segmentation, long-document benchmark |",
            "mental_model": "```text\nsentences -> adjacent similarity -> low score? break : merge -> chunks -> retrieval evaluation\n```",
            "setup_code": r'''
from pathlib import Path
import re
from rag_101 import TfidfVectorizer, find_repo_root

REPO_ROOT = find_repo_root()
TEXT_PATH = REPO_ROOT / "08-advanced-chunking-and-preprocessing/langchain_intro.txt"
text = TEXT_PATH.read_text(encoding="utf-8").strip()
sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
sentences
''',
            "how_it_works": "We fit one vectorizer across sentences, compute adjacent cosine similarity, and start a new chunk when the score falls below a visible threshold. The threshold is a parameter to evaluate, not a universal constant.",
            "baseline_text": "The baseline groups two consecutive sentences per chunk. It is deterministic but ignores topic changes.",
            "baseline_code": r'''
baseline_chunks = [" ".join(sentences[index:index + 2]) for index in range(0, len(sentences), 2)]
baseline_chunks
''',
            "technique_text": "TF-IDF is lexical rather than truly semantic, but it makes breakpoint mechanics inspectable and offline. A production semantic model can replace the vectors while the policy and checks stay the same.",
            "technique_code": r'''
vectorizer = TfidfVectorizer().fit(sentences)
vectors = vectorizer.transform(sentences)
def dot(left, right): return sum(a * b for a, b in zip(left, right, strict=True))
adjacent_scores = [dot(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)]
threshold = 0.05
semantic_chunks = [[sentences[0]]]
for index, sentence in enumerate(sentences[1:]):
    if adjacent_scores[index] < threshold:
        semantic_chunks.append([sentence])
    else:
        semantic_chunks[-1].append(sentence)
semantic_chunks = [" ".join(chunk) for chunk in semantic_chunks]
list(zip(range(1, len(adjacent_scores) + 1), adjacent_scores)), semantic_chunks
''',
            "experiment_text": "We compare whether chunks mix the two known topics (`LangChain` and `Paris/France`) and verify lossless sentence coverage. This fixture is intentionally tiny so every boundary is auditable.",
            "experiment_code": r'''
def mixed_topic_count(chunks: list[str]) -> int:
    source_terms = ("langchain", "llm", "chain", "agent", "retriever")
    destination_terms = ("paris", "france", "eiffel")
    return sum(
        any(term in chunk.lower() for term in source_terms)
        and any(term in chunk.lower() for term in destination_terms)
        for chunk in chunks
    )

experiment_result = {
    "baseline_chunks": len(baseline_chunks),
    "semantic_chunks": len(semantic_chunks),
    "baseline_mixed_topics": mixed_topic_count(baseline_chunks),
    "semantic_mixed_topics": mixed_topic_count(semantic_chunks),
    "break_before_paris": adjacent_scores[2] < threshold,
}
experiment_result
''',
            "evaluation": "The fixed two-sentence baseline mixes topics once because sentence 4 (Paris) is paired with sentence 3 (retrievers). The similarity policy breaks before Paris and produces no mixed-topic chunk, but it also over-segments the LangChain discussion because adjacent sentences 2 and 3 share no terms after preprocessing. This is a useful failure case: lexical breakpoints are not semantic understanding.",
            "checks_code": r'''
assert [sentence for chunk in semantic_chunks for sentence in re.split(r"(?<=[.!?])\s+", chunk)] == sentences
assert experiment_result["baseline_mixed_topics"] == 1
assert experiment_result["semantic_mixed_topics"] == 0
assert experiment_result["break_before_paris"]
print("Semantic chunking checks passed.")
''',
            "decision_guide": "| Situation | Strategy |\n|---|---|\n| Stable headings/sections | Structure-aware first |\n| Short uniform prose | Token/recursive baseline |\n| Topic shifts inside long prose | Evaluated semantic breakpoints |\n| Need parent context | Small-to-big/parent retrieval |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Too many tiny chunks | Threshold too high | Plot score distribution and tune on labels |\n| Topics still mix | Weak representation | Better model or structure signals |\n| Updates rewrite all IDs | Boundary instability | Version chunks and use durable anchors |\n| Retrieval improves but answers worsen | Context lost | Measure support coverage, not chunk purity alone |",
            "production_notes": "### Observability\nTrack chunk counts, size distribution, breakpoint scores, model/version, and retrieval metrics.\n\n### Safety and Guardrails\nDo not send sensitive documents to an unapproved embedding service. Preserve source permissions on every chunk.\n\n### Latency and Cost\nSemantic policies embed sentences and may be much costlier than structural or token splitting; cache by content hash.",
            "practice": "Add a bridging sentence that mentions both LangChain and Paris. Predict how each strategy changes and whether a mixed chunk is actually wrong.",
            "recall": "Toggle - Recall: What does the threshold control?\nThe trade-off between merging coherent neighbors and creating smaller chunks.\n\nToggle - Recall: What must semantic chunking preserve?\nComplete source coverage, order, provenance, and answer-support context.",
            "sources": "- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)\n- Repository fixture: `langchain_intro.txt`",
            "confidence": "High for the transparent fixture experiment",
            "next_review": "Compare neural breakpoints on the shared golden corpus",
        },
    )


if __name__ == "__main__":
    build_embedding_fundamentals()
    build_provider_lesson()
    build_semantic_chunking()
