# Production RAG Learning 101 — Progress Tracker

Last updated: 2026-09-24

## Status legend

| Status | Meaning |
|---|---|
| `Done` | Meets the current deliverable and has been reviewed |
| `In progress` | Active work is underway |
| `Ready` | Scoped and unblocked; can be selected next |
| `Planned` | Accepted backlog item, not yet prepared |
| `Blocked` | Cannot proceed until the recorded dependency is resolved |

Action labels:

- **Enrich** — working example exists; add the full teaching/evaluation layer.
- **Rewrite** — retain the idea but substantially restructure the notebook.
- **Consolidate** — merge overlapping notebooks into one canonical lesson.
- **Split** — divide an oversized notebook into focused lessons.
- **Add** — new material required to complete the curriculum.

## Dashboard

| Workstream | Done | In progress | Remaining | Current note |
|---|---:|---:|---:|---|
| Repository discovery and planning | 3 | 0 | 0 | Inventory, plan, and tracker created |
| Phase 0: backbone and reproducibility | 3 | 1 | 4 | Template/setup complete; static validation added |
| Phase 1: foundations and baseline | 0 | 0 | 6 | Must precede technique comparisons |
| Existing notebook reconstruction | 0 | 0 | 33 | 32 unique; one exact duplicate |
| New production learning modules | 0 | 0 | 9 | Evaluation through deployment/capstone |
| Final clean-run and editorial review | 0 | 0 | 2 | Repository-wide gates |

The counts above track deliverables, not percentages. They change when work is
resized or split.

## Completed planning work

- [x] Inventory repository directories, notebooks, data, and support files.
- [x] Assess notebook explanation depth, imports, outputs, file references, and
  exact duplicates.
- [x] Define the learning architecture, notebook contract, phases, and quality
  gate in the learning plan.

## Phase 0 — Backbone and reproducibility

| ID | Deliverable | Status | Notes |
|---|---|---|---|
| P0-01 | Expand root README into a curriculum map | Ready | Initial links exist; detailed map comes after canonical sequence decision |
| P0-01A | Define target folder structure and migration map | Done | See `docs/TARGET_REPOSITORY_STRUCTURE.md`; no notebooks moved or deleted yet |
| P0-02 | Create canonical notebook template | Done | Template contains the learning flow, experiments, evaluation, production notes, practice, and recall |
| P0-03 | Add environment/setup guide and `.env.example` | Done | Canonical Python version, credentials, kernel, execution rules, and troubleshooting documented |
| P0-04 | Add repository-root path/config helper | Ready | Remove notebook-CWD assumptions |
| P0-05 | Add notebook lint/smoke validation | In progress | Static structure/hygiene validator and three unit tests pass; execution smoke runner remains |
| P0-06 | Align direct dependencies, lock file, imports, and kernel metadata | Planned | Requires clean-run compatibility audit |
| P0-07 | Remove tracked junk and consolidate exact duplicates | Ready | Includes `.DS_Store`, Word lock file, duplicate `2-ReAct_1.ipynb` |

## Phase 1 — Foundations and baseline

| ID | Deliverable | Status | Depends on |
|---|---|---|---|
| FND-01 | Start-here notebook: RAG mental model and architecture | Ready | P0-02 |
| FND-02 | Minimal framework-free retrieval example | Planned | FND-01 |
| FND-03 | Canonical end-to-end LangChain baseline | Planned | FND-02, P0-06 |
| FND-04 | Shared small corpus with stable document/chunk IDs | Planned | P0-04 |
| FND-05 | Golden questions with expected source documents | Planned | FND-04 |
| FND-06 | Baseline retrieval/answer/latency scorecard | Planned | FND-03, FND-05 |

## Existing notebook reconstruction backlog

### Data ingestion and parsing

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| ING-01 | `05-DataIngestParsing/1-dataingestion.ipynb` | Rewrite | P0 | Planned | Make it the ingestion overview; move deep chunking to the chunking stage |
| ING-02 | `05-DataIngestParsing/2-dataparsingpdf.ipynb` | Enrich | P1 | Planned | Text vs scanned PDFs, layout/tables, OCR, parser comparison, quality checks |
| ING-03 | `05-DataIngestParsing/3-dataparsingdoc.ipynb` | Rewrite | P1 | Planned | Explain loader choice, structure preservation, metadata, and failure cases |
| ING-04 | `05-DataIngestParsing/4-csvexcelparsing.ipynb` | Enrich | P1 | Planned | Row/document grain, schema, nulls, formulas, and retrieval implications |
| ING-05 | `05-DataIngestParsing/5-jsonparsing.ipynb` | Enrich | P1 | Planned | JSON/JSONL paths, nested structures, metadata, malformed records |
| ING-06 | `05-DataIngestParsing/6-databaseparsing.ipynb` | Enrich | P1 | Planned | Snapshot vs live SQL retrieval, row grain, provenance, and access safety |
| ING-07 | `05-DataIngestParsing/7-markdownparser.ipynb` | Rewrite | P0 | Planned | Add all missing prose; replace 600+ KB saved output with focused examples |

### Embeddings, chunking, and vector storage

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| EMB-01 | `06-Vector-embedding-and-vector-databases/embedding.ipynb` | Enrich | P0 | Planned | Geometry, cosine/dot/L2, normalization, domain fit, batching, evaluation |
| EMB-02 | `06-Vector-embedding-and-vector-databases/openaiembeddings.ipynb` | Rewrite | P1 | Planned | Provider-neutral baseline, model configuration, compact plots/outputs, cost notes |
| CHK-01 | `08-advanced-chunking-and-preprocessing/semanti_chunking.ipynb` | Enrich | P0 | Planned | Compare against token/recursive baselines and measure retrieval impact |

### Retrieval and reranking

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| RET-01 | `09-hybrid-search-strategies/1-densesparse.ipynb` | Enrich | P0 | Planned | Explain BM25/dense scoring, fusion, normalization, and evaluate both branches |
| RET-02 | `09-hybrid-search-strategies/2-reranking.ipynb` | Enrich | P0 | Planned | Candidate depth, cross-encoder behavior, metrics, latency, and failure modes |
| RET-03 | `09-hybrid-search-strategies/3-mmr.ipynb` | Rewrite | P1 | Planned | Show relevance/diversity objective and compare `lambda_mult` settings |

### Query transformation

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| QRY-01 | `10-query-enhancement/1-query_expansion.ipynb` | Enrich | P1 | Planned | Expansion strategies, query drift, deduplication, and retrieval measurement |
| QRY-02 | `10-query-enhancement/2-query_decomposition.ipynb` | Enrich | P1 | Planned | Atomic subqueries, multi-hop aggregation, parallelism, and completeness |
| QRY-03 | `10-query-enhancement/3-HyDE.ipynb` | Rewrite | P1 | Planned | Clean generated artifacts/outputs; explain domain mismatch and compare baseline |

### Multimodal RAG

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| MM-01 | `11-multiModal-multi-modal-rag/multimodalopenai.ipynb` | Rewrite | P1 | Planned | Add modality mental model, extraction/fusion choices, citations, and evaluation |

### LangGraph and agentic foundations

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| AGF-01 | `section-14-agents-architecture/streaming.ipynb` | Enrich | P1 | Planned | State vs token streaming, sync/async choices, backpressure, cancellation |
| AGF-02 | `section-14-agents-architecture/ReActAgents.ipynb` | Enrich | P1 | Planned | Tool loop, state, termination, errors, memory boundary, observability |
| AGR-01 | `Section-15-agentic-rag/1-agenticrag.ipynb` | Consolidate | P1 | Planned | Merge the best graded/rewrite workflow with the simpler introduction |
| AGR-02 | `Section-15-agentic-rag/1-agenticrag_1.ipynb` | Consolidate | P1 | Planned | Source for the canonical Agentic RAG lesson; retire after verification |
| AGR-03 | `Section-15-agentic-rag/2-ReAct.ipynb` | Rewrite | P1 | Planned | Canonical retrieval-tool ReAct lesson with tool contracts and loop budget |
| AGR-04 | `Section-15-agentic-rag/2-ReAct_1.ipynb` | Consolidate | P0 | Planned | Exact duplicate; remove after canonical notebook is validated |

### Autonomous and multi-agent patterns

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| AUT-01 | `section-16-autonomous-rag/3-COTRag.ipynb` | Rewrite | P1 | Planned | Reframe as explicit planning/retrieval steps; avoid reliance on hidden reasoning |
| AUT-02 | `section-16-autonomous-rag/4-Selfreflection.ipynb` | Rewrite | P1 | Planned | Rubric-based critique, bounded revisions, judge limitations, missing input fix |
| AUT-03 | `section-16-autonomous-rag/5-QueryPlanningdecomposition.ipynb` | Enrich | P2 | Planned | Distinguish from basic decomposition; add orchestration and synthesis checks |
| AUT-04 | `section-16-autonomous-rag/6-Iterativeretrieval.ipynb` | Rewrite | P0 | Planned | Validate graph routing; add termination, retry budget, and missing input fix |
| AUT-05 | `section-16-autonomous-rag/7-answersynthesis.ipynb` | Rewrite | P1 | Planned | Provenance-aware merge, conflict handling, parallel retrieval, missing inputs |
| MAG-01 | `section-17-multi-agents-rags/8-multiagent.ipynb` | Split | P2 | Planned | Create focused network, supervisor, and hierarchical-team notebooks |

### Corrective, adaptive, memory, and cache patterns

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| ADV-01 | `section-18-corrective-rag/2-CorrectiveRAG.ipynb` | Rewrite | P1 | Planned | Add CRAG mental model, graders, thresholds, fallback policy, evaluation |
| ADV-02 | `section-19-adaptive-rag/adaptive-rag.ipynb` | Rewrite | P1 | Planned | Add routing model, branch evaluation, loop safety, and cost trade-offs |
| MEM-01 | `section-20-rag-with-persistant-memory/ragmemory.ipynb` | Rewrite | P1 | Planned | Separate conversation memory, checkpoints, long-term memory, privacy, expiry |
| CAG-01 | `section-21-cache-rag/cache_augment_generation.ipynb` | Rewrite | P1 | Planned | Separate CAG/prompt caching from semantic answer caching and invalidation |

## New modules required for production coverage

| ID | Module | Action | Priority | Status | Outcome |
|---|---|---|---|---|---|
| NEW-01 | Vector indexes and metadata filtering | Add | P1 | Planned | Choose distance/index/filter strategy and understand recall/latency trade-offs |
| NEW-02 | Grounded generation, abstention, and citations | Add | P0 | Planned | Produce answers whose claims can be traced and verified |
| NEW-03 | RAG evaluation and golden datasets | Add | P0 | Planned | Measure retrieval and generation independently |
| NEW-04 | Data and index lifecycle | Add | P1 | Planned | IDs, deduplication, versioning, incremental updates, deletion, re-embedding |
| NEW-05 | Observability and quality monitoring | Add | P1 | Planned | Trace each stage and monitor latency, cost, drift, and answer quality |
| NEW-06 | Security, privacy, and tenant isolation | Add | P0 | Planned | Defend retrieval and generation boundaries in untrusted environments |
| NEW-07 | Performance, cost, caching, and fallbacks | Add | P1 | Planned | Set budgets and design predictable degradation paths |
| NEW-08 | Testing and deployment | Add | P1 | Planned | Unit, integration, smoke, regression, concurrency, and release checks |
| NEW-09 | Production RAG capstone | Add | P2 | Planned | Integrate the curriculum into one measured reference system |

## Known cleanup and validation queue

- [ ] Check every notebook from repository root and from its own directory; make
  one documented execution location canonical.
- [ ] Fix missing example references such as `internal_docs.txt` and ambiguous
  relative paths before calling those notebooks runnable.
- [ ] Replace machine-specific paths and warnings stored in outputs.
- [ ] Remove large full-document dumps from saved outputs.
- [ ] Normalize direct imports (`langchain_core`, `langchain_community`, provider
  packages, and text splitters) after version compatibility is verified.
- [ ] Reconcile Python 3.12/3.13 notebook metadata with the declared environment.
- [ ] Make provider/model choices configurable; identify stale or unavailable
  model identifiers during online validation.
- [ ] Add missing direct dependencies instead of relying on transitive installs.
- [ ] Normalize naming and spelling only with a path migration map
  (`persistent`, `semantic`, consistent section casing).
- [ ] Verify graph topology and termination in every looping LangGraph notebook.

## Review log

| Date | Change | Result |
|---|---|---|
| 2026-09-24 | Initial repository inventory | 33 notebooks, 32 unique, broad advanced coverage, limited explanation/evaluation |
| 2026-09-24 | Learning plan and tracker created | Curriculum sequence, notebook contract, phased backlog, and quality gate established |
| 2026-09-24 | Target repository structure defined | Added canonical curriculum tree and safe migration rules |
| 2026-09-24 | WP-001 course backbone completed | Added start-here guide, setup, credential template, canonical notebook template, validator, and tests |

## Next recommended slice

Complete `P0-02` through `P0-04`, then build `FND-01` through `FND-06` as one
vertical slice. That creates the standard, shared data, baseline, and metrics
needed to improve every existing notebook consistently.
