"""Transparent retrieval and answer metrics used by the foundation notebooks."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Callable, Iterable, Sequence

from .datasets import GoldenQuestion


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]).intersection(relevant_ids)))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]).intersection(relevant)) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(ranked_ids, start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def _answer_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_answer_tokens(prediction))
    expected = Counter(_answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(
    questions: Iterable[GoldenQuestion],
    retrieve_ids: Callable[[str, int], Sequence[str]],
    *,
    k: int = 3,
) -> dict[str, float | int]:
    answerable = [question for question in questions if question.relevant_doc_ids]
    ranked = [retrieve_ids(question.question, k) for question in answerable]
    return {
        "retrieval_questions": len(answerable),
        f"hit_rate@{k}": mean(
            hit_rate_at_k(ids, question.relevant_doc_ids, k)
            for ids, question in zip(ranked, answerable, strict=True)
        ),
        f"recall@{k}": mean(
            recall_at_k(ids, question.relevant_doc_ids, k)
            for ids, question in zip(ranked, answerable, strict=True)
        ),
        "mrr": mean(
            reciprocal_rank(ids, question.relevant_doc_ids)
            for ids, question in zip(ranked, answerable, strict=True)
        ),
    }


def evaluate_answers(
    questions: Iterable[GoldenQuestion],
    answer: Callable[[str], dict[str, object]],
) -> dict[str, float | int]:
    materialized = list(questions)
    predictions = [answer(question.question) for question in materialized]
    answerable_pairs = [
        (question, prediction)
        for question, prediction in zip(materialized, predictions, strict=True)
        if not question.should_abstain
    ]
    abstention_pairs = [
        (question, prediction)
        for question, prediction in zip(materialized, predictions, strict=True)
        if question.should_abstain
    ]
    return {
        "answer_questions": len(materialized),
        "mean_token_f1": mean(
            token_f1(str(prediction["answer"]), question.reference_answer)
            for question, prediction in answerable_pairs
        ),
        "citation_accuracy": mean(
            float(prediction.get("citation") in question.relevant_doc_ids)
            for question, prediction in answerable_pairs
        ),
        "abstention_accuracy": mean(
            float(bool(prediction.get("abstained")))
            for _, prediction in abstention_pairs
        )
        if abstention_pairs
        else 0.0,
    }
