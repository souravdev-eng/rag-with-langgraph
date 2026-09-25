#!/usr/bin/env python3
"""Execute canonical notebook code cells top-to-bottom as a smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from types import ModuleType
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SKIP_TAGS = frozenset({"manual", "online", "paid"})


@dataclass(frozen=True)
class SmokeResult:
    path: Path
    executed_cells: int
    skipped_cells: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def run_notebook(
    path: Path, skip_tags: frozenset[str] = DEFAULT_SKIP_TAGS
) -> SmokeResult:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return SmokeResult(path, 0, 0, f"{type(exc).__name__}: {exc}")

    module_name = f"__notebook_smoke_{abs(hash(path.resolve()))}__"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    namespace = module.__dict__
    sys.modules[module_name] = module
    executed = 0
    skipped = 0

    try:
        for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            tags = frozenset(cell.get("metadata", {}).get("tags", []))
            if tags.intersection(skip_tags):
                skipped += 1
                continue
            try:
                source = _source_text(cell)
                exec(compile(source, f"{path}:cell-{cell_number}", "exec"), namespace)
                executed += 1
            except Exception as exc:  # noqa: BLE001 - report notebook failure verbatim
                return SmokeResult(
                    path,
                    executed,
                    skipped,
                    f"cell {cell_number}: {type(exc).__name__}: {exc}",
                )
    finally:
        sys.modules.pop(module_name, None)

    return SmokeResult(path, executed, skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--run-all-tags",
        action="store_true",
        help="Also execute cells tagged manual, online, or paid.",
    )
    args = parser.parse_args()
    skip_tags = frozenset() if args.run_all_tags else DEFAULT_SKIP_TAGS

    results = [run_notebook(path, skip_tags) for path in args.notebooks]
    for result in results:
        summary = f"executed={result.executed_cells} skipped={result.skipped_cells}"
        if result.passed:
            print(f"PASS {result.path} ({summary})")
        else:
            print(f"FAIL {result.path} ({summary})")
            print(f"  - {result.error}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
