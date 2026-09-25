# Target Repository Structure

This is the target structure for Production RAG Learning 101. It separates the
learning sequence, reusable code, datasets, tests, and documentation.

```text
RAG/
├── 00-start-here/
│   ├── README.md
│   └── 01-how-to-use-this-repo.ipynb
├── 01-rag-foundations/
│   ├── 01-rag-mental-model.ipynb
│   ├── 02-minimal-rag-from-scratch.ipynb
│   └── 03-end-to-end-baseline.ipynb
├── 02-data-ingestion/
│   ├── 01-ingestion-overview.ipynb
│   ├── 02-pdf-parsing.ipynb
│   ├── 03-word-parsing.ipynb
│   ├── 04-csv-excel-parsing.ipynb
│   ├── 05-json-parsing.ipynb
│   ├── 06-database-ingestion.ipynb
│   └── 07-markdown-parsing.ipynb
├── 03-chunking/
│   ├── 01-fixed-token-recursive-chunking.ipynb
│   └── 02-semantic-chunking.ipynb
├── 04-embeddings-and-indexes/
│   ├── 01-embedding-fundamentals.ipynb
│   ├── 02-embedding-providers.ipynb
│   └── 03-vector-indexes-and-filters.ipynb
├── 05-retrieval-and-reranking/
│   ├── 01-dense-vs-sparse.ipynb
│   ├── 02-hybrid-search.ipynb
│   ├── 03-mmr.ipynb
│   └── 04-reranking.ipynb
├── 06-query-transformation/
│   ├── 01-query-expansion.ipynb
│   ├── 02-query-decomposition.ipynb
│   └── 03-hyde.ipynb
├── 07-grounded-generation/
│   ├── 01-context-and-prompt-assembly.ipynb
│   └── 02-citations-and-abstention.ipynb
├── 08-rag-evaluation/
│   ├── 01-golden-datasets.ipynb
│   ├── 02-retrieval-evaluation.ipynb
│   └── 03-answer-evaluation.ipynb
├── 09-multimodal-rag/
│   └── 01-pdf-text-image-rag.ipynb
├── 10-langgraph-and-agentic-rag/
│   ├── 01-langgraph-state-and-streaming.ipynb
│   ├── 02-react-and-tools.ipynb
│   ├── 03-agentic-rag.ipynb
│   ├── 04-self-reflection-and-iterative-retrieval.ipynb
│   ├── 05-corrective-rag.ipynb
│   ├── 06-adaptive-rag.ipynb
│   └── 07-multi-agent-rag.ipynb
├── 11-production-rag/
│   ├── 01-memory-and-persistence.ipynb
│   ├── 02-caching.ipynb
│   ├── 03-data-and-index-lifecycle.ipynb
│   ├── 04-observability.ipynb
│   ├── 05-security-and-access-control.ipynb
│   ├── 06-performance-and-cost.ipynb
│   └── 07-testing-and-deployment.ipynb
├── 12-capstone/
│   └── production-rag-capstone.ipynb
├── src/rag_101/
│   ├── config.py
│   ├── paths.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── evaluation.py
│   └── observability.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── assets/
│   ├── diagrams/
│   └── images/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── notebook-smoke/
├── docs/
├── scripts/
├── .env.example
├── pyproject.toml
└── README.md
```

## Folder rules

- Numbered folders define the learning order.
- Each notebook teaches one primary concept and uses a numbered, lowercase,
  kebab-case name.
- Lesson folders contain notebooks and short local navigation only—not copied
  datasets or reusable application code.
- `src/rag_101/` contains code reused by three or more notebooks.
- `data/raw/` holds immutable source examples; generated chunks and indexes go
  under `data/processed/` and are ignored when appropriate.
- `data/evaluation/` contains golden questions, expected sources, and scorecard
  fixtures.
- `assets/` contains diagrams and learning images referenced by notebooks.
- `tests/` separates fast unit checks, integration checks, and notebook smoke
  execution.
- Secrets never enter the repository; `.env.example` lists variable names only.

## Current-to-target migration map

| Current location | Target location | Migration action |
|---|---|---|
| `05-DataIngestParsing/` | `02-data-ingestion/` | Move notebooks after path/data fixes; centralize sample data |
| `06-Vector-embedding-and-vector-databases/` | `04-embeddings-and-indexes/` | Move and split provider/index concepts clearly |
| `08-advanced-chunking-and-preprocessing/` | `03-chunking/` | Move semantic lesson; add baseline chunking lesson |
| `09-hybrid-search-strategies/` | `05-retrieval-and-reranking/` | Separate dense/sparse basics from hybrid fusion |
| `10-query-enhancement/` | `06-query-transformation/` | Rename consistently and move after validation |
| `11-multiModal-multi-modal-rag/` | `09-multimodal-rag/` | Rewrite and move |
| `section-14-agents-architecture/` | `10-langgraph-and-agentic-rag/` | Use as LangGraph/agent foundations |
| `Section-15-agentic-rag/` | `10-langgraph-and-agentic-rag/` | Consolidate overlapping Agentic RAG and ReAct lessons |
| `section-16-autonomous-rag/` | `10-langgraph-and-agentic-rag/` | Merge related reflection/planning lessons where useful |
| `section-17-multi-agents-rags/` | `10-langgraph-and-agentic-rag/` | Split oversized notebook before moving |
| `section-18-corrective-rag/` | `10-langgraph-and-agentic-rag/` | Rewrite and move |
| `section-19-adaptive-rag/` | `10-langgraph-and-agentic-rag/` | Rewrite and move |
| `section-20-rag-with-persistant-memory/` | `11-production-rag/` | Correct naming and clarify memory boundaries |
| `section-21-cache-rag/` | `11-production-rag/` | Separate caching concepts before moving |

## Safe migration process

For each folder batch:

1. Create the target lesson and update its paths/imports.
2. Run the target notebook and its smoke checks.
3. Compare its learning content and behavior with the original.
4. Update README and tracker links.
5. Move the original to `archive/` temporarily if it still has unmatched value.
6. Remove an archived duplicate only after explicit approval.

No existing notebook will be deleted merely to make the tree look cleaner.
