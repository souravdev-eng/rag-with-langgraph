#!/usr/bin/env python3
"""Rebuild the canonical ingestion overview and Markdown parsing lessons."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
KERNEL = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str, *tags: str):
    cell = nbformat.v4.new_code_cell(text.strip() + "\n")
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def write(path: str, cells: list) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": KERNEL,
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / path)


def build_standard_lesson(path: str, spec: dict[str, str]) -> None:
    """Build the shared 16-section lesson shape from topic-specific content."""
    write(
        path,
        [
            md(
                f"""
# {spec['title']}

| Field | Value |
|---|---|
| Stage | {spec.get('stage', 'Data foundation')} |
| Difficulty | {spec['difficulty']} |
| Status | Complete |
| Requires network/API | No |
| Last reviewed | 2026-09-25 |

Callout - Key idea:
{spec['key_idea']}

## 30-Second Summary

{spec['summary']}

## Why This Matters

{spec['why']}

## Scope

{spec['scope']}
"""
            ),
            md(f"## Mental Model\n\n{spec['mental_model']}"),
            code(spec["setup_code"], "setup"),
            md(f"## How It Works\n\n{spec['how_it_works']}"),
            md(f"## Baseline\n\n{spec['baseline_text']}"),
            code(spec["baseline_code"], "baseline"),
            md(f"## Technique Implementation\n\n{spec['technique_text']}"),
            code(spec["technique_code"], "technique"),
            md(f"## Controlled Experiment\n\n{spec['experiment_text']}"),
            code(spec["experiment_code"], "experiment"),
            md(f"## Evaluation\n\n{spec['evaluation']}"),
            code(spec["checks_code"], "checks"),
            md(f"## Decision Guide\n\n{spec['decision_guide']}"),
            md(f"## Failure Modes and Debugging\n\n{spec['failure_modes']}"),
            md(f"## Production Notes\n\n{spec['production_notes']}"),
            md(
                f"""
## Practice

{spec['practice']}

## Recall

{spec['recall']}

## Sources

{spec['sources']}

## Review Log

| Date | Status | Confidence | Next review focus |
|---|---|---|---|
| 2026-09-25 | Complete; executed and visually reviewed | {spec['confidence']} | {spec['next_review']} |
"""
            ),
        ],
    )


def build_ingestion_overview() -> None:
    write(
        "05-DataIngestParsing/1-dataingestion.ipynb",
        [
            md(
                """
# Data Ingestion: From Files to Trustworthy Documents

| Field | Value |
|---|---|
| Stage | Data foundation |
| Difficulty | Beginner to intermediate |
| Status | Complete |
| Requires network/API | No |
| Last reviewed | 2026-09-25 |

Callout - Key idea:
Ingestion is a data-contract problem, not merely a file-reading problem. A useful document keeps readable content, stable identity, provenance, and validation evidence together.

## 30-Second Summary

This notebook ingests repository-owned UTF-8 text files. It contrasts a text-only baseline with a small, explicit document contract containing a stable content ID, source path, media type, encoding, and size. A controlled quality check shows which downstream requirements the contract satisfies.

## Why This Matters

Retrieval failures often begin before chunking or embeddings: a file is decoded incorrectly, duplicated, detached from its source, or accepted while empty. If ingestion drops identity and provenance, later citations, updates, deletions, and debugging become guesswork.

## Scope

| Covers | Does not cover |
|---|---|
| Local text discovery, decoding, stable IDs, metadata, validation, bounded previews | OCR, layout extraction, deep chunking, remote connectors, access-control implementation |
"""
            ),
            md(
                """
## Mental Model

```text
source bytes -> discover -> decode -> normalize -> validate -> Document contract
                                                        |-> content
                                                        |-> stable ID
                                                        |-> provenance metadata
                                                        `-> quality evidence
```

The source file remains the authority. The ingested document is a traceable representation of that source, not an anonymous string. Chunking comes later and should inherit the document ID and source metadata.
"""
            ),
            code(
                """
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")


REPO_ROOT = find_repo_root()
TEXT_DIR = REPO_ROOT / "05-DataIngestParsing/data/text_files"
SOURCE_FILES = sorted(TEXT_DIR.glob("*.txt"))

assert SOURCE_FILES, f"No text fixtures found under {TEXT_DIR}"
[path.relative_to(REPO_ROOT).as_posix() for path in SOURCE_FILES]
""",
                "setup",
            ),
            md(
                """
## How It Works

1. **Discover** only the intended file type in a bounded directory.
2. **Decode** with an explicit encoding; do not depend on a machine default.
3. **Normalize** line endings and surrounding whitespace without rewriting meaning.
4. **Identify** content with a deterministic hash so identical bytes can be recognized.
5. **Describe** provenance using repository-relative paths and a declared media type.
6. **Validate** required fields before indexing.

The content hash is useful for deduplication, but it is not a complete business identifier. Production systems often combine a source-system ID, version, and content checksum.
"""
            ),
            md(
                """
## Baseline

The baseline reads each file into a string. It proves that bytes can be decoded, but it throws away the source-to-document relationship. Once strings are mixed together, a retriever cannot reliably cite, update, or delete their origins.
"""
            ),
            code(
                """
baseline_documents = [path.read_text(encoding="utf-8") for path in SOURCE_FILES]
baseline_preview = [
    {"characters": len(text), "preview": text[:70].replace("\\n", " ") + "..."}
    for text in baseline_documents
]
baseline_preview
""",
                "baseline",
            ),
            md(
                """
## Technique Implementation

`IngestedDocument` is deliberately small. Its fields are portable across frameworks, while `metadata` can grow with source-specific facts. Repository-relative paths make saved outputs reproducible and avoid leaking a developer's home directory.
"""
            ),
            code(
                """
@dataclass(frozen=True)
class IngestedDocument:
    id: str
    content: str
    metadata: dict[str, Any]


def ingest_text(path: Path) -> IngestedDocument:
    raw = path.read_bytes()
    text = raw.decode("utf-8").replace("\\r\\n", "\\n").strip()
    relative_source = path.resolve().relative_to(REPO_ROOT).as_posix()
    return IngestedDocument(
        id=f"sha256:{sha256(raw).hexdigest()}",
        content=text,
        metadata={
            "source": relative_source,
            "media_type": "text/plain",
            "encoding": "utf-8",
            "bytes": len(raw),
        },
    )


documents = [ingest_text(path) for path in SOURCE_FILES]
[
    {"id": doc.id[:19] + "...", **doc.metadata, "preview": doc.content[:55] + "..."}
    for doc in documents
]
""",
                "technique",
            ),
            md(
                """
## Controlled Experiment

Both approaches read the same two files with the same UTF-8 decoder. We score each representation against four downstream requirements: non-empty content, stable identity, source provenance, and explicit decoding metadata. This is a contract-coverage check, not a retrieval-quality metric.
"""
            ),
            code(
                """
QUALITY_FIELDS = ("content", "stable_id", "source", "encoding")


def baseline_quality(text: str) -> dict[str, bool]:
    return {
        "content": bool(text.strip()),
        "stable_id": False,
        "source": False,
        "encoding": False,
    }


def contract_quality(document: IngestedDocument) -> dict[str, bool]:
    return {
        "content": bool(document.content),
        "stable_id": document.id.startswith("sha256:"),
        "source": bool(document.metadata.get("source")),
        "encoding": document.metadata.get("encoding") == "utf-8",
    }


quality_results = {
    "text_only": sum(sum(baseline_quality(text).values()) for text in baseline_documents)
    / (len(baseline_documents) * len(QUALITY_FIELDS)),
    "document_contract": sum(sum(contract_quality(doc).values()) for doc in documents)
    / (len(documents) * len(QUALITY_FIELDS)),
}
quality_results
""",
                "experiment",
            ),
            md(
                """
## Evaluation

The text-only baseline covers content but none of the operational fields, so it scores **0.25**. The explicit contract covers all four requirements and scores **1.00** for these fixtures. This does not prove the parser handles arbitrary encodings, binary files, or malformed input; it proves the documented local contract.

| Check | Text only | Document contract |
|---|---:|---:|
| Non-empty content | Yes | Yes |
| Stable ID | No | Yes |
| Source provenance | No | Yes |
| Encoding recorded | No | Yes |
"""
            ),
            code(
                """
assert quality_results == {"text_only": 0.25, "document_contract": 1.0}
assert len({document.id for document in documents}) == len(documents)
assert all(not document.metadata["source"].startswith("/") for document in documents)
assert all(document.metadata["bytes"] > 0 for document in documents)
print(f"Ingestion checks passed for {len(documents)} documents.")
""",
                "checks",
            ),
            md(
                """
## Decision Guide

| Source | Preferred ingestion path | Important evidence |
|---|---|---|
| Plain text under your control | Explicit local reader | Encoding, checksum, source path |
| Many files with common rules | Directory/connector loader plus your validation wrapper | Per-file failures and counts |
| PDF or Word | Format-aware parser | Pages/sections, tables, extraction method |
| Database | Bounded query or snapshot | Query/version, row grain, primary key |
| Remote knowledge system | Authorized connector | Source ID, permissions, modified time |

Use a framework loader when it saves format-specific work, but normalize its output into one repository-owned contract before downstream processing.
"""
            ),
            md(
                """
## Failure Modes and Debugging

| Symptom | Likely cause | Verify | Fix |
|---|---|---|---|
| Garbled characters | Wrong decoder | Compare bytes and declared encoding | Detect or configure encoding; quarantine uncertain files |
| Duplicate search results | Same source ingested more than once | Compare content hashes and source IDs | Deduplicate and make ingestion idempotent |
| Citation cannot be resolved | Source metadata was dropped | Trace one result back to its file | Enforce required provenance fields |
| Empty documents reach the index | Parser returned success without content | Count empty/short outputs | Reject, alert, and preserve the failure reason |
| Updates create stale copies | IDs or versions are unstable | Re-ingest one changed source | Define deterministic IDs and replacement semantics |
"""
            ),
            md(
                """
## Production Notes

### Observability
Record discovered, accepted, rejected, duplicate, and changed counts; parser version; duration; and failure category. Log IDs and safe metadata rather than full sensitive content.

### Safety and Guardrails
Treat every source as untrusted input. Bound file size, validate type, scan where required, preserve access-control metadata, and never let a loader silently broaden its authorized scope.

### Latency and Cost
Hashing and parsing are usually linear in input size. Incremental ingestion should skip unchanged sources and isolate retries so one bad file does not restart an entire collection.
"""
            ),
            md(
                """
## Practice

Add a third UTF-8 fixture with Windows line endings, ingest it twice, and verify that your chosen identity policy behaves as intended. Then add an empty file and decide whether to reject or retain it with an explicit status.

## Recall

Toggle - Recall: Why is a string not yet a trustworthy document?
It lacks stable identity, provenance, and validation evidence needed for citations and lifecycle operations.

Toggle - Recall: What does a content hash prove?
It identifies the exact bytes hashed; it does not prove source ownership, permission, freshness, or semantic equivalence.

Toggle - Recall: Why use repository-relative source paths?
They remain portable and avoid exposing machine-specific home directories.

Toggle - Recall: Where should chunking occur?
After source-level ingestion and validation, with document identity and provenance inherited by every chunk.

## Sources

- [Python documentation: Reading and writing files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [Python documentation: `hashlib`](https://docs.python.org/3/library/hashlib.html)
- Repository-owned fixtures under `05-DataIngestParsing/data/text_files/`

## Review Log

| Date | Status | Confidence | Next review focus |
|---|---|---|---|
| 2026-09-25 | Complete; executed and visually reviewed | High for local UTF-8 ingestion | Add source-version and rejection fixtures |
"""
            ),
        ],
    )


def build_markdown_parser() -> None:
    write(
        "05-DataIngestParsing/7-markdownparser.ipynb",
        [
            md(
                """
# Structure-Aware Markdown Parsing

| Field | Value |
|---|---|
| Stage | Data foundation |
| Difficulty | Intermediate |
| Status | Complete |
| Requires network/API | No |
| Last reviewed | 2026-09-25 |

Callout - Key idea:
Markdown headings are retrieval metadata. Preserving their hierarchy creates smaller, explainable sections without discarding the document context needed for citations.

## 30-Second Summary

This notebook compares one-document-per-file loading with a small structure-aware parser. The parser emits sections carrying source, title, heading breadcrumb, and deterministic chunk ID. It validates coverage by reconstructing every source from its parsed sections and checks that retrieval returns focused evidence.

## Why This Matters

Flattening Markdown throws away an inexpensive semantic signal: headings tell us what text is about and where it belongs. Splitting only by character count can mix topics or detach a paragraph from its heading, making results harder to rank and cite.

## Scope

| Covers | Does not cover |
|---|---|
| ATX headings (`#` through `######`), heading hierarchy, preamble handling, stable IDs, lexical comparison | Full CommonMark AST, Setext headings, HTML blocks, semantic chunking, link crawling |
"""
            ),
            md(
                """
## Mental Model

```text
Markdown file -> lines -> heading stack -> section text + breadcrumb
                                            |-> stable section ID
                                            |-> source path
                                            `-> searchable focused unit
```

A heading opens a section. A heading of the same or higher level closes the previous branch; lower-level headings extend the breadcrumb. The section ID combines source identity with its ordinal and breadcrumb so duplicates remain traceable.
"""
            ),
            code(
                """
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import re


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")


REPO_ROOT = find_repo_root()
MARKDOWN_DIR = REPO_ROOT / "05-DataIngestParsing/data/markdown"
SOURCE_FILES = sorted(MARKDOWN_DIR.glob("*.md"))

assert SOURCE_FILES, f"No Markdown fixtures found under {MARKDOWN_DIR}"
len(SOURCE_FILES), [path.name for path in SOURCE_FILES]
""",
                "setup",
            ),
            md(
                """
## How It Works

The parser scans lines once while maintaining a stack of `(level, heading)` pairs. When a heading appears, the current section is emitted, headings at the same or deeper level are removed, and the new heading is pushed. Body lines accumulate until the next heading or end of file.

The implementation preserves the full section text, including its heading. It does not attempt to interpret fenced code, tables, or links; those remain source text. For production-grade CommonMark coverage, use an AST parser and keep this notebook as the behavioral reference.
"""
            ),
            md(
                """
## Baseline

The baseline creates one searchable item per file. This preserves source provenance but makes large files broad: a query may retrieve the correct file while returning far more context than the answer needs.
"""
            ),
            code(
                """
baseline_documents = [
    {
        "id": path.stem,
        "source": path.relative_to(REPO_ROOT).as_posix(),
        "content": path.read_text(encoding="utf-8").strip(),
    }
    for path in SOURCE_FILES
]
[(item["id"], len(item["content"])) for item in baseline_documents]
""",
                "baseline",
            ),
            md(
                """
## Technique Implementation

`MarkdownSection` makes the parent file and heading breadcrumb first-class. Empty heading-only sections are omitted, while a preamble before the first heading receives the breadcrumb `Preamble`. Section ordinals keep IDs unique when a document repeats a heading.
"""
            ),
            code(
                """
@dataclass(frozen=True)
class MarkdownSection:
    id: str
    source: str
    breadcrumb: tuple[str, ...]
    content: str


HEADING = re.compile(r"^(#{1,6})\\s+(.+?)\\s*$")


def parse_markdown(path: Path) -> list[MarkdownSection]:
    source = path.resolve().relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    sections: list[MarkdownSection] = []

    def emit() -> None:
        content = "\\n".join(body).strip()
        if not content:
            return
        breadcrumb = tuple(title for _, title in stack) or ("Preamble",)
        ordinal = len(sections)
        identity = f"{source}|{ordinal}|{' > '.join(breadcrumb)}"
        sections.append(
            MarkdownSection(
                id=sha256(identity.encode("utf-8")).hexdigest()[:16],
                source=source,
                breadcrumb=breadcrumb,
                content=content,
            )
        )

    for line in lines:
        match = HEADING.match(line)
        if match:
            emit()
            body = [line]
            level, title = len(match.group(1)), match.group(2).strip()
            stack = [(old_level, old_title) for old_level, old_title in stack if old_level < level]
            stack.append((level, title))
        else:
            body.append(line)
    emit()
    return sections


sections = [section for path in SOURCE_FILES for section in parse_markdown(path)]
[
    {
        "id": section.id,
        "breadcrumb": " > ".join(section.breadcrumb),
        "characters": len(section.content),
    }
    for section in sections[:8]
]
""",
                "technique",
            ),
            md(
                """
## Controlled Experiment

We compare whole-file and section retrieval with the same transparent TF-IDF scorer. The three queries deliberately target narrow concepts. We measure source accuracy and how much context accompanies the top result. Lower top-result character count is a useful focus proxy here, not a universal quality metric.
"""
            ),
            code(
                """
STOP_WORDS = {"a", "an", "and", "are", "as", "for", "from", "how", "in", "is", "of", "on", "the", "to", "what", "with"}


def tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


def tfidf_rank(query: str, items: list[dict]) -> list[dict]:
    document_frequency = Counter()
    for item in items:
        document_frequency.update(set(tokens(item["content"])))
    query_counts = Counter(tokens(query))
    scored = []
    for item in items:
        counts = Counter(tokens(item["content"]))
        shared = set(query_counts).intersection(counts)
        score = sum(
            (1 + math.log(query_counts[term]))
            * (1 + math.log(counts[term]))
            * (math.log((1 + len(items)) / (1 + document_frequency[term])) + 1) ** 2
            for term in shared
        )
        scored.append({**item, "score": score})
    return sorted(scored, key=lambda item: (-item["score"], item["id"]))


section_documents = [
    {
        "id": section.id,
        "source": section.source,
        "breadcrumb": " > ".join(section.breadcrumb),
        "content": section.content,
    }
    for section in sections
]
queries = [
    ("How are Redux and TanStack Query responsibilities separated?", "redux.md"),
    ("Which theme is persisted across sessions?", "theming.md"),
    ("What can the change history rollback?", "history-service.md"),
]

experiment_rows = []
for query, expected_name in queries:
    file_hit = tfidf_rank(query, baseline_documents)[0]
    section_hit = tfidf_rank(query, section_documents)[0]
    experiment_rows.append(
        {
            "query": query,
            "file_correct": file_hit["source"].endswith(expected_name),
            "section_correct": section_hit["source"].endswith(expected_name),
            "file_characters": len(file_hit["content"]),
            "section_characters": len(section_hit["content"]),
            "section": section_hit.get("breadcrumb"),
        }
    )


for row in experiment_rows:
    print(
        f"file_ok={row['file_correct']!s:<5} section_ok={row['section_correct']!s:<5} "
        f"chars={row['file_characters']:>5}->{row['section_characters']:<5} "
        f"section={row['section']}"
    )
""",
                "experiment",
            ),
            md(
                """
## Evaluation

For the three focused questions, the whole-file baseline retrieves the expected source for **2/3** questions, while section retrieval gets **3/3**. The broad Redux file loses one lexical comparison to another large architecture document; the focused section fixes that result and returns much less context with a citation-ready breadcrumb. We also require **lossless line coverage**: concatenating a file's emitted section content must reproduce its non-blank source lines in order.

The character-count reduction is evidence about context focus on these fixtures, not evidence that smaller chunks always improve retrieval. Very small sections can lose definitions, prerequisites, or cross-section relationships.
"""
            ),
            code(
                """
def nonblank_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


for path in SOURCE_FILES:
    parsed = parse_markdown(path)
    reconstructed = "\\n".join(section.content for section in parsed)
    assert nonblank_lines(reconstructed) == nonblank_lines(path.read_text(encoding="utf-8"))

assert sum(row["file_correct"] for row in experiment_rows) == 2
assert all(row["section_correct"] for row in experiment_rows)
assert all(row["section_characters"] < row["file_characters"] for row in experiment_rows)
assert len({section.id for section in sections}) == len(sections)
print(f"Markdown checks passed: {len(SOURCE_FILES)} files -> {len(sections)} sections.")
""",
                "checks",
            ),
            md(
                """
## Decision Guide

| Situation | Representation | Trade-off |
|---|---|---|
| Short, single-topic note | Whole file | Simple, but broad if the note grows |
| Well-structured documentation | Heading-aware sections | Focused retrieval with useful breadcrumbs |
| Long sections containing several ideas | Heading plus token/recursive sub-splitting | More IDs and boundary decisions |
| Weak or inconsistent headings | AST plus semantic/recursive fallback | More parser and evaluation complexity |
| Exact code/API lookup | Preserve fenced blocks with their explanatory heading | Code can dominate lexical scoring |

Choose boundaries by evaluating answer-support coverage and context noise, not by assuming one chunk size is universally best.
"""
            ),
            md(
                """
## Failure Modes and Debugging

| Symptom | Likely cause | Verify | Fix |
|---|---|---|---|
| Text disappears | Parser emits only non-heading bodies or mishandles preamble | Reconstruct source lines | Add coverage assertions and fixtures |
| Breadcrumb is wrong | Heading stack is not trimmed by level | Inspect nested heading transitions | Test level jumps and repeated headings |
| Code block becomes a false heading | Regex ignores fenced-code state | Add a fenced block containing `#` | Use a CommonMark AST parser |
| Tiny sections rank without context | Granularity is too fine | Inspect answer-support spans | Merge children or retrieve parent context |
| IDs change after unrelated edits | Identity depends on ordinal only | Insert an earlier heading and compare | Use source IDs plus durable anchors/versioning |
"""
            ),
            md(
                """
## Production Notes

### Observability
Track sections per file, size distribution, empty sections, parser version, heading-depth anomalies, reconstruction failures, and retrieval results by source and breadcrumb.

### Safety and Guardrails
Markdown can contain HTML, scripts, remote images, and prompt-like instructions. Parse as data, sanitize before rendering, and preserve authorization metadata on every section.

### Latency and Cost
The scan is linear in source length. Embedding cost grows with emitted section count, so measure duplicate context and update churn before choosing very small boundaries.
"""
            ),
            md(
                """
## Practice

Add a fixture containing a preamble, a skipped heading level, a repeated heading, and a fenced code block with a line beginning `#`. Write expected breadcrumbs first, then decide whether to extend this parser or replace it with an AST implementation.

## Recall

Toggle - Recall: Why keep heading breadcrumbs?
They preserve topical context for ranking, display, and citations even when a section is retrieved alone.

Toggle - Recall: What does reconstruction test?
That parsing did not silently drop or reorder non-blank source lines.

Toggle - Recall: Why is smaller not automatically better?
Small sections reduce noise but can remove definitions or dependencies needed to support an answer.

Toggle - Recall: When should this regex parser be replaced?
When full CommonMark behavior, fenced-block awareness, Setext headings, or complex extensions matter.

## Sources

- [CommonMark specification](https://spec.commonmark.org/)
- [Python documentation: Regular expression operations](https://docs.python.org/3/library/re.html)
- Repository-owned fixtures under `05-DataIngestParsing/data/markdown/`

## Review Log

| Date | Status | Confidence | Next review focus |
|---|---|---|---|
| 2026-09-25 | Complete; executed and visually reviewed | High for the documented ATX-heading subset | Add fenced-code and Setext-heading fixtures |
"""
            ),
        ],
    )


def build_pdf_parser() -> None:
    build_standard_lesson(
        "05-DataIngestParsing/2-dataparsingpdf.ipynb",
        {
            "title": "PDF Parsing: Text, Pages, and Extraction Evidence",
            "difficulty": "Intermediate",
            "key_idea": "A PDF is a positioned graphics container, not a guaranteed stream of reading-order text. Keep page provenance and measure extraction coverage before retrieval.",
            "summary": "This notebook extracts the repository-owned *Attention Is All You Need* PDF with `pypdf`. It compares one giant document with page-aware documents, records page and extraction metadata, and validates that every page yields text for this born-digital fixture.",
            "why": "A parser can return text while silently scrambling columns, dropping equations, or producing nothing for scanned pages. Page-level identity makes failures observable and citations resolvable.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Born-digital text extraction, page provenance, coverage checks, focused retrieval | OCR, table reconstruction, figure understanding, legal assessment of document use |",
            "mental_model": "```text\nPDF bytes -> page objects -> extract text -> validate coverage/order -> page documents\n                         empty page? -> OCR/layout fallback queue\n```",
            "setup_code": r'''
from pathlib import Path
import re
from pypdf import PdfReader

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

REPO_ROOT = find_repo_root()
PDF_PATH = REPO_ROOT / "05-DataIngestParsing/data/pdf/attention.pdf"
reader = PdfReader(PDF_PATH)
assert reader.pages
len(reader.pages), PDF_PATH.relative_to(REPO_ROOT).as_posix()
''',
            "how_it_works": "`pypdf` reads the PDF object model and asks each page for a text representation. We normalize whitespace only for display; the page number and source path stay attached. Empty or implausibly short pages are extraction warnings, not evidence that the source has no content.",
            "baseline_text": "The baseline concatenates every page into one string. It is easy to build but makes a narrow query return the entire paper and loses page-level citation boundaries.",
            "baseline_code": r'''
full_text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
baseline_document = {
    "id": "attention-paper",
    "source": PDF_PATH.relative_to(REPO_ROOT).as_posix(),
    "content": full_text,
}
{"pages": len(reader.pages), "characters": len(full_text), "preview": full_text[:160].replace("\n", " ")}
''',
            "technique_text": "The page-aware representation uses one-based page numbers because that is what a reader sees. Extraction method and character count make parser behavior inspectable, while the stable ID supports citations such as `attention.pdf#page=1`.",
            "technique_code": r'''
page_documents = []
source = PDF_PATH.relative_to(REPO_ROOT).as_posix()
for page_number, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()
    page_documents.append({
        "id": f"attention-paper:p{page_number}",
        "source": source,
        "page": page_number,
        "extraction_method": "pypdf-text",
        "characters": len(text),
        "content": text,
    })
[(item["page"], item["characters"]) for item in page_documents]
''',
            "experiment_text": "For a query about the paper's central contribution, both representations contain the answer. A transparent token-overlap scorer should select page 1, while page-aware retrieval returns far less context. We also measure extraction coverage across all pages.",
            "experiment_code": r'''
def token_overlap(query: str, text: str) -> int:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms)

query = "What architecture is based solely on attention mechanisms without recurrence?"
top_page = max(page_documents, key=lambda item: (token_overlap(query, item["content"]), -item["page"]))
experiment_result = {
    "top_page": top_page["page"],
    "page_characters": top_page["characters"],
    "full_document_characters": len(full_text),
    "nonempty_page_rate": sum(bool(item["content"]) for item in page_documents) / len(page_documents),
}
experiment_result
''',
            "evaluation": "The central-contribution query selects page **1**. All 15 pages produce text, so extraction coverage is **100%** for this fixture. Page-level context is materially smaller than the full-paper baseline, but text presence alone does not validate reading order, equations, or figures.",
            "checks_code": r'''
assert experiment_result["top_page"] == 1
assert experiment_result["nonempty_page_rate"] == 1.0
assert experiment_result["page_characters"] < experiment_result["full_document_characters"]
assert [item["page"] for item in page_documents] == list(range(1, len(reader.pages) + 1))
print(f"PDF checks passed for {len(page_documents)} pages.")
''',
            "decision_guide": "| PDF type | First parser | Escalate when |\n|---|---|---|\n| Born-digital prose | Text extractor | Reading order or equations fail |\n| Scanned pages | OCR | Confidence/layout is poor |\n| Tables/forms | Layout-aware parser | Cells or fields are merged |\n| Figures/charts | Multimodal extraction | Meaning is not present in captions |",
            "failure_modes": "| Symptom | Likely cause | Fix |\n|---|---|---|\n| Empty page | Image-only scan | OCR and record confidence |\n| Column text interleaves | Layout lost | Use layout-aware extraction |\n| Header repeats everywhere | Boilerplate retained | Detect repeated marginal text |\n| Citation off by one | Zero/one-based mismatch | Store reader-facing page numbers explicitly |",
            "production_notes": "### Observability\nTrack pages discovered, empty/short pages, characters per page, parser version, OCR use, and rejected files.\n\n### Safety and Guardrails\nBound size and page count, treat embedded content as untrusted, and preserve authorization metadata.\n\n### Latency and Cost\nText extraction is local; OCR and multimodal parsing add substantial compute and should run only on pages that need them.",
            "practice": "Add an image-only one-page PDF fixture. Confirm that the text extractor flags it, then define the metadata an OCR fallback must return.",
            "recall": "Toggle - Recall: Why retain page numbers?\nThey provide citation boundaries and localize extraction failures.\n\nToggle - Recall: Does non-empty text prove correct parsing?\nNo; order, tables, equations, and figures may still be wrong.",
            "sources": "- [pypdf: Extract text from a PDF](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)\n- Repository fixture: `attention.pdf`",
            "confidence": "High for this born-digital fixture",
            "next_review": "Add scanned and multi-column adversarial fixtures",
        },
    )


def build_word_parser() -> None:
    build_standard_lesson(
        "05-DataIngestParsing/3-dataparsingdoc.ipynb",
        {
            "title": "Word Parsing: Preserve Headings and Tables",
            "difficulty": "Intermediate",
            "key_idea": "Word documents contain structural objects. Extract paragraphs, heading hierarchy, and tables with provenance instead of flattening everything into one string.",
            "summary": "This notebook parses a repository-owned proposal with `python-docx`. It compares plain paragraph extraction with typed blocks and verifies that five headings and one table remain identifiable.",
            "why": "Flattening a DOCX can detach values from table headers and erase section boundaries. Structure-aware blocks support better chunking, citations, and quality checks.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Paragraph styles, headings, table rows, source metadata | Tracked changes, comments, floating text boxes, images, legacy `.doc` |",
            "mental_model": "```text\nDOCX package -> paragraphs + styles + tables -> typed blocks -> validated documents\n```",
            "setup_code": r'''
from pathlib import Path
from docx import Document

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file(): return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

REPO_ROOT = find_repo_root()
DOCX_PATH = REPO_ROOT / "05-DataIngestParsing/data/word_files/proposal.docx"
document = Document(DOCX_PATH)
len(document.paragraphs), len(document.tables)
''',
            "how_it_works": "A DOCX file is an Open Packaging Convention archive. `python-docx` exposes paragraphs with style names and tables with rows/cells. We convert those objects into bounded records while keeping the source and ordinal.",
            "baseline_text": "The baseline joins non-empty paragraphs. It preserves readable prose but omits table content and does not label headings.",
            "baseline_code": r'''
baseline_paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
baseline_text = "\n".join(baseline_paragraphs)
{"paragraphs": len(baseline_paragraphs), "characters": len(baseline_text), "preview": baseline_text[:120]}
''',
            "technique_text": "Typed blocks keep each paragraph's style and each table row's header-to-value mapping. This is still an extraction layer; downstream chunking can group blocks under the nearest heading.",
            "technique_code": r'''
source = DOCX_PATH.relative_to(REPO_ROOT).as_posix()
blocks = []
for ordinal, paragraph in enumerate(document.paragraphs):
    text = paragraph.text.strip()
    if text:
        blocks.append({
            "id": f"proposal:p{ordinal}", "source": source, "kind": "paragraph",
            "style": paragraph.style.name, "content": text,
        })
for table_index, table in enumerate(document.tables):
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    for row_index, row in enumerate(table.rows[1:], start=1):
        values = [cell.text.strip() for cell in row.cells]
        record = dict(zip(headers, values, strict=True))
        blocks.append({
            "id": f"proposal:t{table_index}:r{row_index}", "source": source,
            "kind": "table_row", "record": record,
            "content": "; ".join(f"{key}: {value}" for key, value in record.items()),
        })
[(block["kind"], block.get("style"), block["content"][:65]) for block in blocks]
''',
            "experiment_text": "We test structural coverage: heading blocks should match the source's Heading styles, and table rows should remain queryable as header-value records. The baseline cannot satisfy either requirement.",
            "experiment_code": r'''
heading_blocks = [block for block in blocks if block.get("style", "").startswith("Heading")]
table_blocks = [block for block in blocks if block["kind"] == "table_row"]
experiment_result = {
    "baseline_has_heading_labels": False,
    "baseline_has_table_rows": False,
    "heading_count": len(heading_blocks),
    "table_row_count": len(table_blocks),
    "typed_block_count": len(blocks),
}
experiment_result
''',
            "evaluation": "The typed representation preserves **5 heading blocks** and the rows from the document's single table; the paragraph-only baseline preserves neither signal. The check covers this fixture, not every object a DOCX can contain.",
            "checks_code": r'''
assert experiment_result["heading_count"] == 5
assert experiment_result["table_row_count"] > 0
assert all(block["source"] == source for block in blocks)
assert any("Budget" in block["content"] or "budget" in block["content"] for block in blocks)
print(f"Word checks passed for {len(blocks)} typed blocks.")
''',
            "decision_guide": "| Need | Approach |\n|---|---|\n| Simple prose | Paragraph extraction with styles |\n| Tables | Header-aware row records |\n| Exact visual fidelity | Render plus layout-aware extraction |\n| Legacy `.doc` | Convert in a controlled pipeline first |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Table values lose meaning | Headers dropped | Map headers to every row |\n| Sections merge | Styles ignored | Preserve heading level/style |\n| Text is missing | Content lives in shapes/headers | Add object-specific extraction and fixtures |\n| Duplicate rows | Merged cells expanded | Detect spans and normalize carefully |",
            "production_notes": "### Observability\nTrack paragraph, heading, table, row, and empty-block counts by parser version.\n\n### Safety and Guardrails\nTreat macros, links, and embedded objects as untrusted; this lesson reads `.docx` content only.\n\n### Latency and Cost\nLocal XML parsing is inexpensive; rendering, OCR, and layout models are slower fallbacks.",
            "practice": "Add a merged-cell table and a Heading 2 subsection. Specify the expected records before changing the parser.",
            "recall": "Toggle - Recall: Why keep paragraph styles?\nThey expose section hierarchy without guessing from font size.\n\nToggle - Recall: Why turn table rows into mappings?\nHeaders give each value meaning and improve retrieval/citation context.",
            "sources": "- [python-docx user guide](https://python-docx.readthedocs.io/en/latest/)\n- Repository fixture: `proposal.docx`",
            "confidence": "High for paragraphs and simple tables",
            "next_review": "Add merged cells and non-body object fixtures",
        },
    )


def build_structured_parser() -> None:
    build_standard_lesson(
        "05-DataIngestParsing/4-csvexcelparsing.ipynb",
        {
            "title": "CSV and Excel Parsing: Choose the Right Row Grain",
            "difficulty": "Beginner to intermediate",
            "key_idea": "A table is not one blob. Preserve schema and row identity so filtering, retrieval, and citations operate at the intended grain.",
            "summary": "This notebook loads equivalent product data from CSV and Excel, validates schema and value parity, then builds one retrieval document per row with typed metadata.",
            "why": "Flattening a spreadsheet hides columns, nulls, and row identity. Row-aware documents support exact filters and explain which record supplied an answer.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Schema checks, null checks, CSV/Excel parity, row documents | Formulas, merged cells, multi-sheet business logic, live spreadsheet sessions |",
            "mental_model": "```text\nCSV/XLSX -> dataframe -> validate schema/types/nulls -> row records -> text + filter metadata\n```",
            "setup_code": r'''
from pathlib import Path
import pandas as pd

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file(): return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "05-DataIngestParsing/data/structured_files"
CSV_PATH, XLSX_PATH = DATA_DIR / "products.csv", DATA_DIR / "inventory.xlsx"
csv_frame = pd.read_csv(CSV_PATH)
xlsx_frame = pd.read_excel(XLSX_PATH)
csv_frame
''',
            "how_it_works": "We declare the expected columns, load each format with an explicit engine path, validate shape/nulls/types, and only then serialize rows. Numeric values remain typed metadata even though the retrieval text is human-readable.",
            "baseline_text": "The baseline serializes the entire table as one document. It is easy to preview but gives a retriever one oversized result and no row-level citation.",
            "baseline_code": r'''
baseline_documents = [
    {"source": path.relative_to(REPO_ROOT).as_posix(), "content": frame.to_string(index=False)}
    for path, frame in ((CSV_PATH, csv_frame), (XLSX_PATH, xlsx_frame))
]
[(item["source"], len(item["content"])) for item in baseline_documents]
''',
            "technique_text": "Each row becomes a document keyed by source and row number. Product/category/price/stock stay available for filters, while labeled text provides a lexical or embedding representation.",
            "technique_code": r'''
EXPECTED_COLUMNS = ["Product", "Category", "Price", "Stock", "Description"]

def row_documents(path: Path, frame: pd.DataFrame) -> list[dict]:
    source = path.relative_to(REPO_ROOT).as_posix()
    docs = []
    for row_number, row in frame.iterrows():
        record = row.to_dict()
        docs.append({
            "id": f"{path.stem}:row:{row_number + 2}", "source": source,
            "row_number": row_number + 2, "metadata": record,
            "content": " | ".join(f"{column}: {record[column]}" for column in EXPECTED_COLUMNS),
        })
    return docs

csv_documents = row_documents(CSV_PATH, csv_frame)
xlsx_documents = row_documents(XLSX_PATH, xlsx_frame)
[(doc["id"], doc["content"]) for doc in csv_documents]
''',
            "experiment_text": "The CSV and Excel files are intended to represent the same five records. We verify schema, nulls, and value parity before comparing whole-table context with the focused row returned for an RGB mechanical keyboard query.",
            "experiment_code": r'''
pd.testing.assert_frame_equal(csv_frame, xlsx_frame, check_dtype=False)
query_terms = {"mechanical", "keyboard", "rgb"}
top_row = max(csv_documents, key=lambda doc: len(query_terms & set(doc["content"].lower().replace("|", " ").split())))
experiment_result = {
    "rows": len(csv_frame),
    "columns": len(csv_frame.columns),
    "null_cells": int(csv_frame.isna().sum().sum()),
    "top_product": top_row["metadata"]["Product"],
    "row_characters": len(top_row["content"]),
    "table_characters": len(baseline_documents[0]["content"]),
}
experiment_result
''',
            "evaluation": "Both formats contain the same **5 × 5** values with no null cells. The focused query retrieves the `Keyboard` row, whose context is smaller and carries exact price/stock metadata. Real workbooks need sheet, formula, date, and merged-cell policies.",
            "checks_code": r'''
assert list(csv_frame.columns) == EXPECTED_COLUMNS
assert experiment_result["rows"] == 5 and experiment_result["columns"] == 5
assert experiment_result["null_cells"] == 0
assert experiment_result["top_product"] == "Keyboard"
assert experiment_result["row_characters"] < experiment_result["table_characters"]
print("Structured-file checks passed for CSV and Excel.")
''',
            "decision_guide": "| Question grain | Document grain |\n|---|---|\n| Product lookup | One row |\n| Category summary | Grouped aggregate plus source rows |\n| Narrative notes sheet | Section/paragraph |\n| Exact numeric filtering | Typed database/dataframe filter before semantic ranking |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Prices become text | Type inference/configuration | Validate dtypes and units |\n| Rows cite wrong source | Row IDs omitted | Store source, sheet, and row |\n| Formulas look stale | Cached values used | Define recalculation policy |\n| Null becomes `nan` text | Serialization before validation | Handle missingness explicitly |",
            "production_notes": "### Observability\nRecord file/sheet, row/column counts, schema drift, nulls, duplicates, and coercion failures.\n\n### Safety and Guardrails\nFormula cells can contain external links; never execute spreadsheet formulas during ingestion.\n\n### Latency and Cost\nFilter and aggregate structured fields before embedding; do not pay to encode values that exact predicates answer better.",
            "practice": "Add a duplicate product and a null price. Define whether to reject, deduplicate, or retain each row before changing code.",
            "recall": "Toggle - Recall: Why keep numbers typed?\nExact filters and comparisons are safer than asking embeddings to represent arithmetic.\n\nToggle - Recall: What determines document grain?\nThe questions and lifecycle operations the system must support.",
            "sources": "- [pandas I/O documentation](https://pandas.pydata.org/docs/user_guide/io.html)\n- Repository fixtures: `products.csv`, `inventory.xlsx`",
            "confidence": "High for the single-sheet fixtures",
            "next_review": "Add formula, date, null, and multi-sheet cases",
        },
    )


def build_json_parser() -> None:
    build_standard_lesson(
        "05-DataIngestParsing/5-jsonparsing.ipynb",
        {
            "title": "JSON and JSONL Parsing: Preserve Record Boundaries",
            "difficulty": "Intermediate",
            "key_idea": "Choose records with an explicit JSON path and stable IDs; do not flatten an entire nested payload into an untraceable string.",
            "summary": "This notebook parses nested employee JSON and line-delimited event JSONL using the standard library. It emits one employee or event record per document and validates IDs, line provenance, and malformed-line isolation.",
            "why": "Nested arrays describe different grains, while JSONL is designed for independent records. Treating both as one blob makes updates, errors, and citations unnecessarily broad.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Nested paths, JSONL lines, record IDs, typed metadata | Streaming huge files, schema registries, arbitrary flattening, remote APIs |",
            "mental_model": "```text\nJSON  -> select $.employees[*] -> employee documents\nJSONL -> parse each line        -> event documents (line provenance)\n```",
            "setup_code": r'''
from pathlib import Path
import json

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file(): return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

REPO_ROOT = find_repo_root()
JSON_DIR = REPO_ROOT / "05-DataIngestParsing/data/json_files"
COMPANY_PATH, EVENTS_PATH = JSON_DIR / "company_data.json", JSON_DIR / "events.jsonl"
company_payload = json.loads(COMPANY_PATH.read_text(encoding="utf-8"))
len(company_payload["employees"]), company_payload["company"]
''',
            "how_it_works": "For nested JSON we select a named array and build IDs from source-owned keys. For JSONL we parse and validate one line at a time, so one malformed record can be reported without losing the rest of the file.",
            "baseline_text": "The baseline keeps each entire file as raw text. It preserves bytes but mixes employee, department, and event grains into broad retrieval units.",
            "baseline_code": r'''
baseline_documents = [
    {"source": path.relative_to(REPO_ROOT).as_posix(), "content": path.read_text(encoding="utf-8")}
    for path in (COMPANY_PATH, EVENTS_PATH)
]
[(item["source"], len(item["content"])) for item in baseline_documents]
''',
            "technique_text": "Employee documents retain skills and project lists; event documents retain typed fields plus the one-based source line. Unknown event fields remain in metadata rather than being silently discarded.",
            "technique_code": r'''
company_source = COMPANY_PATH.relative_to(REPO_ROOT).as_posix()
employee_documents = []
for employee in company_payload["employees"]:
    projects = ", ".join(f"{item['name']} ({item['status']})" for item in employee["projects"])
    employee_documents.append({
        "id": f"employee:{employee['id']}", "source": company_source,
        "metadata": employee,
        "content": f"Name: {employee['name']} | Role: {employee['role']} | Skills: {', '.join(employee['skills'])} | Projects: {projects}",
    })

event_documents, malformed_lines = [], []
events_source = EVENTS_PATH.relative_to(REPO_ROOT).as_posix()
for line_number, line in enumerate(EVENTS_PATH.read_text(encoding="utf-8").splitlines(), start=1):
    try:
        event = json.loads(line)
        event_documents.append({
            "id": f"event:{line_number}", "source": events_source,
            "line": line_number, "metadata": event,
            "content": " | ".join(f"{key}: {value}" for key, value in event.items()),
        })
    except json.JSONDecodeError as error:
        malformed_lines.append({"line": line_number, "error": str(error)})

[(doc["id"], doc["content"]) for doc in employee_documents + event_documents]
''',
            "experiment_text": "We compare raw-file context with the employee record answering a question about the Data Scientist's skills and projects. We also verify that every JSONL line produces exactly one event document and no hidden parse failures.",
            "experiment_code": r'''
top_employee = next(doc for doc in employee_documents if doc["metadata"]["role"] == "Data Scientist")
experiment_result = {
    "employee_records": len(employee_documents),
    "event_records": len(event_documents),
    "malformed_lines": len(malformed_lines),
    "answer_name": top_employee["metadata"]["name"],
    "record_characters": len(top_employee["content"]),
    "raw_company_characters": len(baseline_documents[0]["content"]),
}
experiment_result
''',
            "evaluation": "The nested payload yields **2 employee records** and the JSONL file yields **3 event records**, with no malformed lines. The Data Scientist record resolves to Jane Smith and is much smaller than the full company payload.",
            "checks_code": r'''
assert experiment_result == {
    "employee_records": 2, "event_records": 3, "malformed_lines": 0,
    "answer_name": "Jane Smith", "record_characters": experiment_result["record_characters"],
    "raw_company_characters": experiment_result["raw_company_characters"],
}
assert len({doc["id"] for doc in employee_documents + event_documents}) == 5
assert experiment_result["record_characters"] < experiment_result["raw_company_characters"]
print("JSON checks passed for nested JSON and JSONL.")
''',
            "decision_guide": "| Shape | Grain |\n|---|---|\n| Array of entities | One document per entity |\n| JSONL event stream | One document per line/event |\n| Nested configuration | Section by stable path |\n| Frequently filtered fields | Keep typed metadata; filter before ranking |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| One bad line kills batch | Whole-file JSONL parse | Parse/isolate by line |\n| IDs change | Array index used as identity | Use source keys plus version |\n| Nested facts vanish | Over-aggressive flattening | Define and test selected paths |\n| Numbers/dates become ambiguous | String-only serialization | Preserve typed metadata/schema |",
            "production_notes": "### Observability\nTrack records, malformed lines, missing keys, schema versions, unknown fields, and duplicates.\n\n### Safety and Guardrails\nBound nesting depth and record size; redact sensitive fields before logs or embeddings.\n\n### Latency and Cost\nStream large JSONL files and checkpoint progress rather than loading them entirely.",
            "practice": "Append one malformed JSONL line and one valid event with a new field. Verify isolation and decide whether schema drift is accepted or quarantined.",
            "recall": "Toggle - Recall: Why is JSON path selection important?\nIt defines the entity grain and prevents unrelated nested objects from becoming one blob.\n\nToggle - Recall: Why keep JSONL line numbers?\nThey locate failures and support source-level citations.",
            "sources": "- [Python `json` documentation](https://docs.python.org/3/library/json.html)\n- [JSON Lines format](https://jsonlines.org/)\n- Repository JSON fixtures",
            "confidence": "High for the documented fixture shapes",
            "next_review": "Add malformed, deeply nested, and schema-drift fixtures",
        },
    )


def build_database_parser() -> None:
    build_standard_lesson(
        "05-DataIngestParsing/6-databaseparsing.ipynb",
        {
            "title": "Database Ingestion: Snapshot Rows with Provenance",
            "difficulty": "Intermediate",
            "key_idea": "Database ingestion needs an explicit query, row grain, key, and snapshot boundary. The connection itself is not a document contract.",
            "summary": "This notebook opens the repository SQLite fixture in read-only mode, inspects schema, and compares raw table dumps with a bounded joined snapshot of projects and their leads.",
            "why": "Unbounded `SELECT *`, unstable row IDs, and many-to-many joins can create stale or duplicated retrieval documents. Query provenance makes the snapshot reproducible and auditable.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Read-only SQLite, schema inspection, bounded join, row IDs, join checks | Live text-to-SQL, CDC, credentials, production database load |",
            "mental_model": "```text\nread-only DB -> inspect schema -> bounded query -> validate grain/join -> row documents + query provenance\n```",
            "setup_code": r'''
from pathlib import Path
import sqlite3

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file(): return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

REPO_ROOT = find_repo_root()
DB_PATH = REPO_ROOT / "05-DataIngestParsing/data/databases/company.db"
connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
tables
''',
            "how_it_works": "We inspect known tables, run a fixed query with explicit columns and ordering, verify the output grain, then serialize each project row with its primary key and lead details. In production, record query/version and snapshot time or change watermark.",
            "baseline_text": "The baseline dumps every row from both tables separately. It preserves rows but forces downstream code to infer the relationship and query provenance.",
            "baseline_code": r'''
baseline_rows = {
    table: [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
    for table in tables
}
{table: len(rows) for table, rows in baseline_rows.items()}
''',
            "technique_text": "The technique uses an explicit one-project-per-row join. Qualified aliases prevent column-name collisions, and the project primary key becomes the document ID.",
            "technique_code": r'''
SNAPSHOT_QUERY = """
SELECT
    p.id AS project_id,
    p.name AS project_name,
    p.status AS project_status,
    p.budget AS project_budget,
    e.id AS lead_id,
    e.name AS lead_name,
    e.role AS lead_role,
    e.department AS lead_department
FROM projects AS p
LEFT JOIN employees AS e ON e.id = p.lead_id
ORDER BY p.id
"""
snapshot_rows = [dict(row) for row in connection.execute(SNAPSHOT_QUERY)]
source = DB_PATH.relative_to(REPO_ROOT).as_posix()
project_documents = [
    {
        "id": f"project:{row['project_id']}", "source": source,
        "metadata": row,
        "content": (
            f"Project: {row['project_name']} | Status: {row['project_status']} | "
            f"Budget: {row['project_budget']} | Lead: {row['lead_name']} ({row['lead_role']})"
        ),
    }
    for row in snapshot_rows
]
[(doc["id"], doc["content"]) for doc in project_documents]
''',
            "experiment_text": "We verify that the left join neither drops nor multiplies projects, every document has a stable primary-key ID, and the active RAG Implementation project resolves to its expected lead.",
            "experiment_code": r'''
project_count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
active_project = next(doc for doc in project_documents if doc["metadata"]["project_status"] == "Active")
experiment_result = {
    "source_projects": project_count,
    "snapshot_rows": len(snapshot_rows),
    "unique_project_ids": len({row["project_id"] for row in snapshot_rows}),
    "active_project": active_project["metadata"]["project_name"],
    "active_lead": active_project["metadata"]["lead_name"],
}
experiment_result
''',
            "evaluation": "The query produces exactly **4 rows for 4 projects** with 4 unique project IDs. The active `RAG Implementation` project resolves to `John Doe`. This validates the fixture join; it does not establish a production refresh or authorization policy.",
            "checks_code": r'''
assert experiment_result == {
    "source_projects": 4, "snapshot_rows": 4, "unique_project_ids": 4,
    "active_project": "RAG Implementation", "active_lead": "John Doe",
}
assert len({doc["id"] for doc in project_documents}) == len(project_documents)
assert connection.execute("PRAGMA query_only").fetchone()[0] == 0  # URI mode enforces read-only at the file layer.
connection.close()
print("Database checks passed for the bounded read-only snapshot.")
''',
            "decision_guide": "| Need | Pattern |\n|---|---|\n| Stable knowledge snapshot | Versioned bounded extract |\n| Frequently changing facts | Live authorized SQL/tool call |\n| Semantic search over text columns | Snapshot rows plus typed metadata |\n| Incremental refresh | CDC/watermark with idempotent upsert/delete |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Duplicate documents | Many-to-many join multiplication | Assert grain and unique keys |\n| Deleted rows remain searchable | Append-only refresh | Propagate tombstones/deletes |\n| Stale answers | Snapshot age hidden | Record freshness/watermark |\n| Sensitive rows leak | Authorization after retrieval | Apply row/tenant policy before candidate selection |",
            "production_notes": "### Observability\nRecord query/version, row counts, unmatched joins, duplicate keys, watermark, duration, and source snapshot ID.\n\n### Safety and Guardrails\nUse least-privilege read credentials, parameterize filters, bound results, and enforce tenant policy at the source.\n\n### Latency and Cost\nPrefer incremental extracts for retrieval indexes; reserve live queries for facts whose freshness justifies database load.",
            "practice": "Insert a fixture project with a missing lead into a copy of the database. Verify left-join behavior and define whether the document is accepted or quarantined.",
            "recall": "Toggle - Recall: What must a database snapshot record?\nQuery/version, grain, primary key, source, and freshness boundary.\n\nToggle - Recall: Why check row counts after a join?\nA join can silently drop or multiply the intended population.",
            "sources": "- [Python `sqlite3` documentation](https://docs.python.org/3/library/sqlite3.html)\n- [SQLite URI filenames](https://www.sqlite.org/uri.html)\n- Repository fixture: `company.db`",
            "confidence": "High for the bounded SQLite fixture",
            "next_review": "Add unmatched and many-to-many join fixtures",
        },
    )

if __name__ == "__main__":
    build_ingestion_overview()
    build_pdf_parser()
    build_word_parser()
    build_structured_parser()
    build_json_parser()
    build_database_parser()
    build_markdown_parser()
