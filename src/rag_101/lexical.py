"""Small, transparent lexical retrieval components for foundation lessons."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .datasets import CorpusDocument


STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer with L2-normalized output vectors."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: tuple[float, ...] = ()

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizer":
        materialized = list(texts)
        document_frequency: Counter[str] = Counter()
        for text in materialized:
            document_frequency.update(set(tokenize(text)))

        terms = sorted(document_frequency)
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        document_count = len(materialized)
        self.idf = tuple(
            math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in terms
        )
        return self

    def transform_one(self, text: str) -> list[float]:
        if not self.vocabulary:
            raise RuntimeError("Call fit before transform_one.")

        counts = Counter(tokenize(text))
        vector = [0.0] * len(self.vocabulary)
        for term, count in counts.items():
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] = (1 + math.log(count)) * self.idf[index]

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def transform(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.transform_one(text) for text in texts]


@dataclass(frozen=True)
class SearchResult:
    document: CorpusDocument
    score: float


class TfidfRetriever:
    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        if not documents:
            raise ValueError("At least one document is required.")
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer().fit(document.content for document in documents)
        self.document_vectors = self.vectorizer.transform(
            document.content for document in documents
        )

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be at least 1.")
        query_vector = self.vectorizer.transform_one(query)
        ranked = [
            SearchResult(document=document, score=cosine_similarity(query_vector, vector))
            for document, vector in zip(
                self.documents, self.document_vectors, strict=True
            )
        ]
        ranked.sort(key=lambda result: (-result.score, result.document.id))
        return ranked[:k]


def extractive_answer(
    question: str,
    results: Sequence[SearchResult],
    *,
    minimum_retrieval_score: float = 0.08,
    minimum_query_coverage: float = 0.45,
) -> dict[str, object]:
    """Select a supporting sentence or abstain; this is not an LLM."""

    if not results or results[0].score < minimum_retrieval_score:
        return {
            "answer": "I don't know based on the provided knowledge base.",
            "citation": None,
            "abstained": True,
        }

    question_terms = set(tokenize(question))
    top_document = results[0].document
    candidates: list[tuple[int, int, str, CorpusDocument]] = []
    sentences = re.split(r"(?<=[.!?])\s+", top_document.content)
    for sentence in sentences:
        overlap = len(question_terms.intersection(tokenize(sentence)))
        candidates.append((overlap, -len(sentence), sentence, top_document))

    overlap, _, sentence, document = max(candidates, key=lambda item: item[:2])
    query_coverage = overlap / len(question_terms) if question_terms else 0.0
    if query_coverage < minimum_query_coverage:
        return {
            "answer": "I don't know based on the provided knowledge base.",
            "citation": None,
            "abstained": True,
        }

    return {
        "answer": sentence,
        "citation": document.id,
        "abstained": False,
    }
