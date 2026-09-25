import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_notebook_smoke import run_notebook


class NotebookSmokeTests(unittest.TestCase):
    def write_notebook(self, cells: list[dict]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "lesson.ipynb"
        path.write_text(
            json.dumps(
                {
                    "cells": cells,
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def code(source: str, *tags: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": list(tags)},
            "outputs": [],
            "source": [source],
        }

    def test_cells_share_state_and_execute_in_order(self):
        path = self.write_notebook(
            [self.code("value = 40"), self.code("assert value + 2 == 42")]
        )

        result = run_notebook(path)

        self.assertTrue(result.passed, result.error)
        self.assertEqual(result.executed_cells, 2)

    def test_failure_reports_cell_number(self):
        path = self.write_notebook([self.code("raise ValueError('broken')")])

        result = run_notebook(path)

        self.assertFalse(result.passed)
        self.assertIn("cell 1: ValueError: broken", result.error or "")

    def test_online_cells_are_skipped_by_default(self):
        path = self.write_notebook(
            [self.code("raise RuntimeError('network')", "online"), self.code("result = 1")]
        )

        result = run_notebook(path)

        self.assertTrue(result.passed, result.error)
        self.assertEqual(result.executed_cells, 1)
        self.assertEqual(result.skipped_cells, 1)


if __name__ == "__main__":
    unittest.main()
