#!/usr/bin/env python3
"""Validate the structure and basic hygiene of canonical learning notebooks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_SECTIONS = (
    "30-Second Summary",
    "Why This Matters",
    "Scope",
    "Mental Model",
    "How It Works",
    "Baseline",
    "Technique Implementation",
    "Controlled Experiment",
    "Evaluation",
    "Decision Guide",
    "Failure Modes and Debugging",
    "Production Notes",
    "Practice",
    "Recall",
    "Sources",
    "Review Log",
)

SUSPICIOUS_SOURCE_PATTERNS = (
    (re.compile(r"(?:sk|gsk)_[A-Za-z0-9_-]{12,}"), "possible API key"),
    (re.compile(r"/Users/[^/\s]+/"), "macOS user-specific absolute path"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s]+"), "Windows user-specific absolute path"),
)


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_notebook(path: Path) -> ValidationResult:
    errors: list[str] = []

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ValidationResult(path, ("file does not exist",))
    except json.JSONDecodeError as exc:
        return ValidationResult(path, (f"invalid JSON: {exc}",))

    if notebook.get("nbformat") != 4:
        errors.append("notebook must use nbformat 4")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        return ValidationResult(path, tuple(errors + ["notebook has no cells"]))

    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]

    if not markdown_cells:
        errors.append("notebook has no markdown cells")
    if not code_cells:
        errors.append("notebook has no code cells")

    markdown = "\n".join(_source_text(cell) for cell in markdown_cells)
    first_markdown = _source_text(markdown_cells[0]).lstrip() if markdown_cells else ""
    if not first_markdown.startswith("# "):
        errors.append("first markdown cell must begin with an H1 title")

    headings = {
        match.group(1).strip()
        for match in re.finditer(r"^#{2,3}\s+(.+?)\s*$", markdown, re.MULTILINE)
    }
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            errors.append(f"missing required section: {section}")

    for index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"code cell {index} contains a saved error output")

    source = "\n".join(_source_text(cell) for cell in cells)
    for pattern, label in SUSPICIOUS_SOURCE_PATTERNS:
        if pattern.search(source):
            errors.append(f"source contains {label}")

    return ValidationResult(path, tuple(errors))


def validate_many(paths: Iterable[Path]) -> list[ValidationResult]:
    return [validate_notebook(path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()

    results = validate_many(args.notebooks)
    for result in results:
        if result.passed:
            print(f"PASS {result.path}")
            continue
        print(f"FAIL {result.path}")
        for error in result.errors:
            print(f"  - {error}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
