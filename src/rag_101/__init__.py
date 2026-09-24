"""Reusable helpers for the Production RAG Learning 101 curriculum."""

from .datasets import CorpusDocument, GoldenQuestion, load_corpus, load_golden_questions
from .evaluation import evaluate_answers, evaluate_retrieval
from .lexical import SearchResult, TfidfRetriever, TfidfVectorizer, extractive_answer
from .paths import find_repo_root, repo_path

__all__ = [
    "CorpusDocument",
    "GoldenQuestion",
    "SearchResult",
    "TfidfRetriever",
    "TfidfVectorizer",
    "evaluate_answers",
    "evaluate_retrieval",
    "extractive_answer",
    "find_repo_root",
    "load_corpus",
    "load_golden_questions",
    "repo_path",
]
