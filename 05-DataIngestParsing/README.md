# Data Ingestion and Parsing

Run these lessons in order. Each uses repository-owned fixtures, requires no
network access, and is saved with verified outputs.

1. [Data ingestion overview](1-dataingestion.ipynb) — document contracts,
   stable IDs, provenance, and validation.
2. [PDF parsing](2-dataparsingpdf.ipynb) — page-aware extraction and coverage.
3. [Word parsing](3-dataparsingdoc.ipynb) — paragraphs, headings, and tables.
4. [CSV and Excel parsing](4-csvexcelparsing.ipynb) — schema checks and row grain.
5. [JSON and JSONL parsing](5-jsonparsing.ipynb) — nested paths and record lines.
6. [Database ingestion](6-databaseparsing.ipynb) — bounded read-only snapshots.
7. [Markdown parsing](7-markdownparser.ipynb) — heading-aware sections and
   lossless coverage checks.

From the repository root, validate and smoke-test the section with:

```bash
uv run python scripts/validate_notebook_structure.py 05-DataIngestParsing/*.ipynb
uv run python scripts/run_notebook_smoke.py 05-DataIngestParsing/*.ipynb
```
