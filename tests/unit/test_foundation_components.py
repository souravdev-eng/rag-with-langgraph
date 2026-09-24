import unittest
from pathlib import Path

from rag_101.datasets import load_corpus, load_golden_questions
from rag_101.evaluation import evaluate_answers, evaluate_retrieval
from rag_101.lexical import TfidfRetriever, extractive_answer
from rag_101.paths import find_repo_root, repo_path


class FoundationComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_corpus()
        cls.questions = load_golden_questions()
        cls.retriever = TfidfRetriever(cls.documents)

    def test_repo_root_and_data_paths_are_stable(self):
        root = find_repo_root(Path(__file__))
        self.assertEqual(root / "pyproject.toml", repo_path("pyproject.toml"))
        self.assertTrue(repo_path("data", "raw", "rag_101_corpus.jsonl").is_file())

    def test_dataset_ids_are_unique_and_references_exist(self):
        document_ids = [document.id for document in self.documents]
        self.assertEqual(len(document_ids), len(set(document_ids)))
        for question in self.questions:
            self.assertTrue(set(question.relevant_doc_ids).issubset(document_ids))

    def test_retriever_finds_every_answerable_document_at_rank_one(self):
        for question in self.questions:
            if not question.relevant_doc_ids:
                continue
            result = self.retriever.search(question.question, k=1)[0]
            self.assertIn(result.document.id, question.relevant_doc_ids)

    def test_baseline_metrics_meet_foundation_thresholds(self):
        retrieval_metrics = evaluate_retrieval(
            self.questions,
            lambda query, k: [
                result.document.id for result in self.retriever.search(query, k)
            ],
            k=3,
        )
        answer_metrics = evaluate_answers(
            self.questions,
            lambda query: extractive_answer(query, self.retriever.search(query, k=3)),
        )

        self.assertEqual(retrieval_metrics["hit_rate@3"], 1.0)
        self.assertEqual(retrieval_metrics["mrr"], 1.0)
        self.assertGreaterEqual(answer_metrics["mean_token_f1"], 0.45)
        self.assertEqual(answer_metrics["citation_accuracy"], 1.0)
        self.assertEqual(answer_metrics["abstention_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
