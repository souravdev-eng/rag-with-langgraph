# Production RAG Learning 101 — Progress Tracker

Last updated: 2026-09-25

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
| Phase 0: backbone and reproducibility | 7 | 0 | 1 | Dependencies, kernels, navigation, and clean-run compatibility aligned; tracked-junk cleanup remains |
| Phase 1: foundations and baseline | 6 | 0 | 0 | Three lessons, shared corpus, golden set, and scorecard complete |
| Existing notebook reconstruction | 33 | 0 | 0 | All existing lessons rebuilt, executed, and visually reviewed |
| New production learning modules | 0 | 0 | 9 | Evaluation through deployment/capstone |
| Final clean-run and editorial review | 2 | 0 | 0 | All 39 course notebooks pass both repository-root and notebook-directory gates |

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
| P0-01 | Expand root README into a curriculum map | Done | Root navigation links every completed lesson in the recommended sequence |
| P0-01A | Define target folder structure and migration map | Done | See `docs/TARGET_REPOSITORY_STRUCTURE.md`; no notebooks moved or deleted yet |
| P0-02 | Create canonical notebook template | Done | Template contains the learning flow, experiments, evaluation, production notes, practice, and recall |
| P0-03 | Add environment/setup guide and `.env.example` | Done | Canonical Python version, credentials, kernel, execution rules, and troubleshooting documented |
| P0-04 | Add repository-root path/config helper | Done | `src/rag_101/paths.py` resolves and optionally validates repository paths |
| P0-05 | Add notebook lint/smoke validation | Done | Static structure/hygiene validation plus tag-aware top-to-bottom execution runner |
| P0-06 | Align direct dependencies, lock file, imports, and kernel metadata | Done | Python 3.12 metadata, direct notebook tooling dependencies, lock file, and clean-run compatibility verified |
| P0-07 | Remove tracked junk and consolidate exact duplicates | Ready | Exact duplicate is a byte-identical compatibility copy; tracked-junk deletion remains a separate cleanup action |

## Phase 1 — Foundations and baseline

| ID | Deliverable | Status | Depends on |
|---|---|---|---|
| FND-01 | Start-here notebook: RAG mental model and architecture | Done | Clear offline/online paths, failure boundaries, citations, and abstention |
| FND-02 | Minimal framework-free retrieval example | Done | Standard-library tokenization, TF-IDF, cosine search, and controlled comparison |
| FND-03 | Canonical end-to-end LangChain baseline | Done | `Document`, `Embeddings`, `InMemoryVectorStore`, and runnable composition |
| FND-04 | Shared small corpus with stable document/chunk IDs | Done | Eight fictional Northstar policy documents with metadata |
| FND-05 | Golden questions with expected source documents | Done | Seven answerable questions plus one abstention case |
| FND-06 | Baseline retrieval/answer/latency scorecard | Done | Retrieval, answer, citation, abstention, parity, and latency signals |

## Existing notebook reconstruction backlog

### Data ingestion and parsing

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| ING-01 | `05-DataIngestParsing/1-dataingestion.ipynb` | Rewrite | P0 | Done | Explicit ingestion contract, stable IDs, provenance, validation, and evaluation |
| ING-02 | `05-DataIngestParsing/2-dataparsingpdf.ipynb` | Enrich | P1 | Done | Page-aware text extraction, coverage checks, OCR/layout decision points |
| ING-03 | `05-DataIngestParsing/3-dataparsingdoc.ipynb` | Rewrite | P1 | Done | Typed paragraph/heading/table blocks with structural checks |
| ING-04 | `05-DataIngestParsing/4-csvexcelparsing.ipynb` | Enrich | P1 | Done | Schema/null/parity checks and row-grain retrieval documents |
| ING-05 | `05-DataIngestParsing/5-jsonparsing.ipynb` | Enrich | P1 | Done | Nested-path and JSONL record parsing with line provenance |
| ING-06 | `05-DataIngestParsing/6-databaseparsing.ipynb` | Enrich | P1 | Done | Read-only bounded snapshot, explicit join grain, and provenance |
| ING-07 | `05-DataIngestParsing/7-markdownparser.ipynb` | Rewrite | P0 | Done | Heading-aware parser, lossless coverage check, and focused retrieval comparison |

### Embeddings, chunking, and vector storage

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| EMB-01 | `06-Vector-embedding-and-vector-databases/embedding.ipynb` | Enrich | P0 | Done | Geometry, dot/cosine/L2, normalization, ranking checks, and production guidance |
| EMB-02 | `06-Vector-embedding-and-vector-databases/openaiembeddings.ipynb` | Rewrite | P1 | Done | Provider-neutral contract, deterministic offline comparison, migration and cost notes |
| CHK-01 | `08-advanced-chunking-and-preprocessing/semanti_chunking.ipynb` | Enrich | P0 | Done | Adjacent-similarity breakpoints, fixed baseline, coverage and over-segmentation checks |

### Retrieval and reranking

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| RET-01 | `09-hybrid-search-strategies/1-densesparse.ipynb` | Enrich | P0 | Done | BM25, lossy dense LSA, equal/weighted RRF, branch metrics, and fusion failure case |
| RET-02 | `09-hybrid-search-strategies/2-reranking.ipynb` | Enrich | P0 | Done | Candidate-depth recall gate, transparent pairwise proxy, MRR, and latency boundary |
| RET-03 | `09-hybrid-search-strategies/3-mmr.ipynb` | Rewrite | P1 | Done | From-scratch MMR, lambda sweep, redundancy metrics, and irrelevant-novelty failure |

### Query transformation

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| QRY-01 | `10-query-enhancement/1-query_expansion.ipynb` | Enrich | P1 | Done | Curated expansion, deduplication, hit@1, and aggressive-drift failure case |
| QRY-02 | `10-query-enhancement/2-query_decomposition.ipynb` | Enrich | P1 | Done | Atomic subqueries, evidence coverage, parallel retrieval, and cited synthesis |
| QRY-03 | `10-query-enhancement/3-HyDE.ipynb` | Rewrite | P1 | Done | Baseline comparison, hypothetical-document retrieval, and strict evidence boundary |

### Multimodal RAG

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| MM-01 | `11-multiModal-multi-modal-rag/multimodalopenai.ipynb` | Rewrite | P1 | Done | Text/image records, bar-geometry check, modality agreement, citations, and numeric abstention |

### LangGraph and agentic foundations

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| AGF-01 | `section-14-agents-architecture/streaming.ipynb` | Enrich | P1 | Done | Values/updates event contracts, reconstruction parity, cancellation/backpressure notes |
| AGF-02 | `section-14-agents-architecture/ReActAgents.ipynb` | Enrich | P1 | Done | Offline typed tool loop, routing, abstention, step budget, and observable trace |
| AGR-01 | `Section-15-agentic-rag/1-agenticrag.ipynb` | Consolidate | P1 | Done | Bounded retrieve-grade-rewrite workflow with citation and abstention |
| AGR-02 | `Section-15-agentic-rag/1-agenticrag_1.ipynb` | Consolidate | P1 | Done | Three-route retrieve/calculate/abstain controller with labeled checks |
| AGR-03 | `Section-15-agentic-rag/2-ReAct.ipynb` | Rewrite | P1 | Done | Canonical typed retrieval-tool loop with citation, abstention, and step budget |
| AGR-04 | `Section-15-agentic-rag/2-ReAct_1.ipynb` | Consolidate | P0 | Done | Byte-identical validated compatibility copy retained pending deletion approval |

### Autonomous and multi-agent patterns

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| AUT-01 | `section-16-autonomous-rag/3-COTRag.ipynb` | Rewrite | P1 | Done | Explicit task plan, evidence slots, coverage gate, and cited synthesis |
| AUT-02 | `section-16-autonomous-rag/4-Selfreflection.ipynb` | Rewrite | P1 | Done | Evidence-backed rubric, one-revision budget, and deterministic acceptance gate |
| AUT-03 | `section-16-autonomous-rag/5-QueryPlanningdecomposition.ipynb` | Enrich | P2 | Done | Typed dependency graph, parallel-ready retrievals, and complete comparison |
| AUT-04 | `section-16-autonomous-rag/6-Iterativeretrieval.ipynb` | Rewrite | P0 | Done | Query history, duplicate guard, success path, and exhausted-budget abstention |
| AUT-05 | `section-16-autonomous-rag/7-answersynthesis.ipynb` | Rewrite | P1 | Done | Effective-date conflict detection, declared resolution, and provenance-aware answer |
| MAG-01 | `section-17-multi-agents-rags/8-multiagent.ipynb` | Split | P2 | Done | Compatibility overview plus focused network, supervisor, and hierarchical-team lessons |

### Corrective, adaptive, memory, and cache patterns

| ID | Notebook | Action | Priority | Status | Primary upgrade |
|---|---|---|---|---|---|
| ADV-01 | `section-18-corrective-rag/2-CorrectiveRAG.ipynb` | Rewrite | P1 | Done | Transparent grader, threshold, one rewrite, bounded fallback, and abstention policy |
| ADV-02 | `section-19-adaptive-rag/adaptive-rag.ipynb` | Rewrite | P1 | Done | Labeled routing, branch contracts, shared budget, and measured cost trade-off |
| MEM-01 | `section-20-rag-with-persistant-memory/ragmemory.ipynb` | Rewrite | P1 | Done | Conversation, checkpoint, and consented long-term memory with TTL/delete checks |
| CAG-01 | `section-21-cache-rag/cache_augment_generation.ipynb` | Rewrite | P1 | Done | Context-cache reuse versus semantic-answer risk with version/as-of invalidation |

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

- [x] Check every notebook from repository root and from its own directory; make
  one documented execution location canonical.
- [x] Fix missing example references such as `internal_docs.txt` and ambiguous
  relative paths before calling those notebooks runnable.
- [x] Replace machine-specific paths and warnings stored in outputs.
- [x] Remove large full-document dumps from saved outputs.
- [x] Normalize direct imports (`langchain_core`, `langchain_community`, provider
  packages, and text splitters) after version compatibility is verified.
- [x] Reconcile Python 3.12/3.13 notebook metadata with the declared environment.
- [x] Make provider/model choices configurable; identify stale or unavailable
  model identifiers during online validation.
- [x] Add missing direct dependencies instead of relying on transitive installs.
- [ ] Normalize naming and spelling only with a path migration map
  (`persistent`, `semantic`, consistent section casing).
- [x] Verify graph topology and termination in every looping LangGraph notebook.

## Review log

| Date | Change | Result |
|---|---|---|
| 2026-09-24 | Initial repository inventory | 33 notebooks, 32 unique, broad advanced coverage, limited explanation/evaluation |
| 2026-09-24 | Learning plan and tracker created | Curriculum sequence, notebook contract, phased backlog, and quality gate established |
| 2026-09-24 | Target repository structure defined | Added canonical curriculum tree and safe migration rules |
| 2026-09-24 | WP-001 course backbone completed | Added start-here guide, setup, credential template, canonical notebook template, validator, and tests |
| 2026-09-24 | WP-002 RAG foundations completed | Added three runnable lessons, shared components, corpus, golden questions, and baseline scorecard |
| 2026-09-25 | ING-01 through ING-07 completed | Rebuilt all ingestion lessons, executed them top-to-bottom, visually reviewed rendered HTML, and reduced saved notebook size/output noise |
| 2026-09-25 | EMB-01, EMB-02, and CHK-01 completed | Added transparent geometry/provider contracts and an evaluated offline semantic-chunking lesson; executed and visually reviewed all three |
| 2026-09-25 | RET-01 through RET-03 completed | Rebuilt dense/sparse/hybrid, reranking, and MMR lessons; retained measured failure cases, executed, and visually reviewed all three |
| 2026-09-25 | QRY-01 through QRY-03 completed | Rebuilt expansion, decomposition, and HyDE with drift, completeness, and hallucinated-hypothesis checks; executed and visually reviewed all three |
| 2026-09-25 | MM-01, AGF-01, and AGF-02 completed | Rebuilt multimodal extraction/fusion and LangGraph streaming/ReAct foundations; executed and visually reviewed all three |
| 2026-09-25 | AGR-01 through AGR-04 completed | Consolidated correction, routing, and retrieval-tool loop lessons; kept the approved-to-retain duplicate path byte-identical after validation |
| 2026-09-25 | AUT-01 through AUT-05 completed | Rebuilt planning, reflection, dependency-aware orchestration, iterative retrieval, and conflict-aware synthesis with bounded control flow |
| 2026-09-25 | MAG-01, ADV-01, ADV-02, MEM-01, and CAG-01 completed | Split multi-agent topologies and rebuilt corrective, adaptive, memory, and cache lessons with explicit failure boundaries |
| 2026-09-25 | Repository-wide clean run completed | All 39 course notebooks passed structure, smoke, root execution, saved-output, and visual-review gates; 10 support tests passed |

## Next recommended slice

Existing notebook reconstruction is complete. The next independent slice is
either the explicitly approved tracked-junk cleanup (`P0-07`) or the separate
new-production-module backlog (`NEW-01` through `NEW-09`).
