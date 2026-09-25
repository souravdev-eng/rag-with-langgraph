"""Typed loaders for the small foundation corpus and golden questions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .paths import repo_path


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    reference_answer: str
    should_abstain: bool = False


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def load_corpus(path: Path | None = None) -> list[CorpusDocument]:
    source = path or repo_path("data", "raw", "rag_101_corpus.jsonl", must_exist=True)
    return [CorpusDocument(**record) for record in _read_jsonl(source)]


def load_golden_questions(path: Path | None = None) -> list[GoldenQuestion]:
    source = path or repo_path(
        "data", "evaluation", "golden_questions.jsonl", must_exist=True
    )
    questions: list[GoldenQuestion] = []
    for record in _read_jsonl(source):
        record["relevant_doc_ids"] = tuple(record["relevant_doc_ids"])
        questions.append(GoldenQuestion(**record))
    return questions
