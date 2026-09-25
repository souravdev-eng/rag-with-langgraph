import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_notebook_structure import REQUIRED_SECTIONS, validate_notebook


def notebook(markdown: str, code: str = "result = 1") -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [markdown]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [code],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class NotebookStructureValidationTests(unittest.TestCase):
    def write_notebook(self, payload: dict) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "lesson.ipynb"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_notebook_passes(self):
        sections = "\n\n".join(f"## {section}" for section in REQUIRED_SECTIONS)
        path = self.write_notebook(notebook(f"# Lesson\n\n{sections}"))

        result = validate_notebook(path)

        self.assertTrue(result.passed, result.errors)

    def test_missing_sections_are_reported(self):
        path = self.write_notebook(notebook("# Lesson\n\n## Scope"))

        result = validate_notebook(path)

        self.assertFalse(result.passed)
        self.assertIn("missing required section: Evaluation", result.errors)

    def test_saved_error_and_absolute_path_are_reported(self):
        payload = notebook("# Lesson")
        payload["cells"][1]["source"] = ["path = '/Users/example/data.txt'"]
        payload["cells"][1]["outputs"] = [
            {"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": []}
        ]
        path = self.write_notebook(payload)

        result = validate_notebook(path)

        self.assertIn("code cell 1 contains a saved error output", result.errors)
        self.assertIn("source contains macOS user-specific absolute path", result.errors)


if __name__ == "__main__":
    unittest.main()
