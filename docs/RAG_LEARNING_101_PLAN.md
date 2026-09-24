# Production RAG Learning 101 — Learning Plan

Status: Proposed working plan  
Created: 2026-09-24  
Scope: turn this bootcamp repository into a self-contained, revision-friendly,
production RAG curriculum.

## North star

Opening any canonical notebook should be enough to understand, run, evaluate,
and explain that RAG technique without needing the original bootcamp lesson.

Callout - Key idea:
The repository will teach a decision process, not a collection of LangChain
snippets: establish a baseline, measure the failure, introduce one technique,
compare the result, and state when the extra complexity is justified.

## What “done” means

The completed repository should let a learner:

1. Explain the end-to-end RAG data and query paths.
2. Build a small baseline without hiding the important mechanics behind a
   framework.
3. Choose ingestion, chunking, embedding, indexing, retrieval, reranking, and
   query-transformation strategies based on evidence.
4. Measure retrieval quality and answer quality separately.
5. Diagnose common failures instead of changing prompts blindly.
6. Add citations, tracing, security boundaries, caching, and data lifecycle
   controls needed by a production system.
7. Explain when agentic RAG is useful and when a deterministic workflow is the
   better design.
8. Re-run every canonical notebook from a clean environment.

## Repository baseline

The initial audit found:

| Area | Current state | Implication |
|---|---|---|
| Breadth | 33 notebooks spanning ingestion through agentic and cache-based RAG | There is strong raw material to preserve |
| Unique notebooks | 32 after one exact duplicate is removed | Duplicate ReAct notebook should be consolidated |
| Teaching depth | 19 notebooks have fewer than 100 markdown words; 15 have fewer than 50 | Most notebooks need a mental model, trade-offs, failure modes, and recall material |
| Saved outputs | About 2.7 MB across notebooks | Large outputs obscure the lesson and can contain machine-specific noise |
| Reproducibility | Multiple kernel versions, mixed import paths, relative-path assumptions, and API-dependent cells | A clean-run contract is required |
| Navigation | Root README previously contained one line; numbering begins at section 05 and later changes case/style | A curriculum map and consistent naming are needed |
| Evaluation | No dedicated evaluation module or golden dataset | Technique comparisons currently cannot prove improvement |
| Production coverage | No dedicated modules for observability, security, access control, index lifecycle, or deployment testing | These must be added for a production-oriented curriculum |
| Existing strengths | Sample files, practical loaders, vector retrieval, reranking, query transformation, multimodal, LangGraph, CRAG, adaptive RAG, memory, caching | These examples form the core of the future curriculum |

The audit was static: notebook structure, explanatory content, imports, local
file references, outputs, and duplicate hashes were inspected. Full clean-room
execution is a separate task because several notebooks require paid APIs or
network services.

## Learning architecture

The learning path is organized by the decisions made in a real RAG system.

```text
Question
  -> query understanding / routing
  -> retrieval (sparse, dense, hybrid)
  -> reranking / context selection
  -> grounded generation + citations
  -> answer evaluation

Source data
  -> parsing
  -> cleaning / metadata
  -> chunking
  -> embeddings
  -> index construction and lifecycle

Both paths
  -> observability, security, cost, latency, tests, and feedback
```

### Curriculum stages

| Stage | Purpose | Existing material | Required additions |
|---|---|---|---|
| 0. Start here | Environment, vocabulary, architecture, how to use the repo | Minimal | Orientation and setup guide |
| 1. Baseline RAG | Build the smallest end-to-end system and establish failure cases | Pieces exist across later notebooks | One canonical baseline notebook |
| 2. Data foundation | Parse, clean, preserve metadata, and handle formats | Strong ingestion section | OCR/layout, metadata contract, deduplication |
| 3. Chunking and embeddings | Understand representation and boundaries | Embedding and semantic chunking notebooks | Token-aware baseline, experiment harness, vector-index concepts |
| 4. Retrieval | Dense, sparse, hybrid, MMR, filters, reranking | Good practical coverage | Quantitative comparison and metadata filtering |
| 5. Query and generation | Expansion, decomposition, HyDE, grounding, citations | Query techniques exist | Grounded generation and citation notebook |
| 6. Evaluation | Prove whether a change improves the system | Missing | Golden set, retrieval metrics, answer metrics, regression tests |
| 7. Multimodal RAG | Retrieve and answer over text and images | One code-heavy notebook | Clear modality model and evaluation |
| 8. Workflow and agentic RAG | Routing, tools, reflection, corrective and adaptive flows | Extensive coverage | Consolidation, stopping rules, budgets, deterministic-vs-agent guidance |
| 9. Production | Operate safely and predictably | Memory and cache examples | Lifecycle, tracing, security, latency/cost, testing, deployment |
| 10. Capstone | Integrate the decisions into one reviewed system | Missing | Production-style reference implementation and scorecard |

## Canonical notebook contract

Every completed technique notebook will follow the same learning rhythm. The
number of sections may shrink for a small topic, but the outcomes may not.

1. **Title and 30-second summary** — the problem, the technique, and the main
   trade-off.
2. **Why it matters** — one realistic scenario that carries through the lesson.
3. **Scope and prerequisites** — what the notebook covers and intentionally
   omits.
4. **Mental model** — a small diagram placed beside the idea it teaches.
5. **Mechanics** — the algorithm or data flow in plain language.
6. **Minimal implementation** — enough from-scratch code to expose the core
   mechanism.
7. **Framework implementation** — a reusable LangChain/LangGraph version when
   the framework adds value.
8. **Controlled experiment** — baseline versus the technique on the same small
   dataset and queries.
9. **Evaluation** — relevant retrieval, generation, latency, and cost signals.
10. **Decision guide** — when to use it, when not to use it, and alternatives.
11. **Failure modes and debugging** — symptoms, likely causes, and fixes.
12. **Production notes** — configuration, observability, safety, and scaling
    implications.
13. **Practice and recall** — one exercise plus short questions answerable aloud.
14. **Sources and review log** — primary documentation/papers and review status.

### Code contract

Canonical notebooks must also satisfy these rules:

- Run top-to-bottom from the repository root using the documented Python
  environment.
- Resolve data paths from the repository, not from the author's current working
  directory.
- Never contain secrets; `.env.example` documents required variables.
- Put paid or network-dependent cells behind an explicit, documented switch and
  provide a local/default learning path where practical.
- Use deterministic seeds where supported and show concise expected outputs.
- Keep notebook outputs small; do not store full documents, embeddings, or long
  traces in committed output cells.
- State model names and important parameters in one configuration cell.
- Use current, consistent import paths and declare every direct dependency.
- Separate reusable implementation code from presentation code once repetition
  appears in three notebooks.
- Include at least one assertion or check that proves the demonstrated behavior.

## Execution plan

### Phase 0 — Course backbone and reproducibility

Deliverables:

- Expand the root README into the curriculum entry point.
- Introduce the target repository structure and migrate existing notebooks in
  small, verified batches without deleting source material.
- Add a notebook template and contribution/checklist guide.
- Add environment setup, `.env.example`, data-path helper, and a lightweight
  notebook validation command.
- Decide the canonical package/import versions, then sync `pyproject.toml`, the
  lock file, and notebook kernels.
- Remove tracked editor/OS artifacts and consolidate exact duplicates.
- Record which notebooks require OpenAI, Groq, Tavily, Hugging Face downloads,
  or other network access.

Exit check: a new learner can install the project and run the local smoke
notebook without guessing about paths or credentials.

### Phase 1 — Foundations and measurable baseline

Deliverables:

- Add `Start Here: How RAG Works` with separate offline indexing and online
  query flows.
- Add a canonical end-to-end baseline over a small repository-owned dataset.
- Add a small golden question set with expected source documents.
- Add retrieval metrics (`hit rate/recall@k`, `MRR`, and optionally `nDCG`) and
  simple answer checks.
- Capture the baseline score, latency, and model/token configuration.

Exit check: later techniques can be compared against a shared baseline instead
of being demonstrated in isolation.

### Phase 2 — Data, chunking, embeddings, and indexes

Deliverables:

- Rebuild the ingestion notebooks around a common `Document` and metadata
  contract.
- Explain parser selection, encoding, OCR/layout/table limitations, and data
  quality checks.
- Compare fixed/token/recursive/semantic chunking on the same documents.
- Deepen embeddings with similarity metrics, normalization, model selection,
  dimensions, batching, and domain mismatch.
- Add vector-index fundamentals, persistence, metadata filters, and index
  lifecycle concepts.
- Add deduplication, document IDs, versioning, incremental updates, and deletion.

Exit check: the learner can explain how an input document becomes a trustworthy,
versioned set of searchable chunks.

### Phase 3 — Retrieval and query strategy

Deliverables:

- Enrich dense, BM25, hybrid, MMR, and reranking notebooks.
- Use the common dataset and evaluation harness for every comparison.
- Add metadata filtering and parent/child or small-to-big retrieval.
- Enrich query expansion, decomposition, and HyDE with routing criteria and
  measurable failure cases.
- Make latency/quality trade-offs visible for first-stage retrieval and
  reranking.

Exit check: each technique earns its added complexity through an observed
retrieval improvement or a clearly documented use case.

### Phase 4 — Grounded generation, citations, and evaluation

Deliverables:

- Add prompt/context assembly, abstention, source attribution, and citation
  verification.
- Separate retrieval failures from generation failures.
- Add answer relevance, groundedness/faithfulness, completeness, and citation
  checks; clearly label heuristic, LLM-judge, and human evaluation.
- Create a regression runner and scorecard that can compare notebook variants.
- Document judge bias, leakage, nondeterminism, and metric limitations.

Exit check: an answer is not called “better” without inspectable evidence.

### Phase 5 — Multimodal, workflow, and agentic patterns

Deliverables:

- Rewrite multimodal RAG around text/image extraction, representation, fusion,
  and modality-aware evaluation.
- Establish LangGraph state, nodes, edges, streaming, checkpoints, retries, and
  interrupts before advanced graphs.
- Consolidate overlapping Agentic RAG and ReAct notebooks.
- Reframe “chain-of-thought RAG” as explicit planning and retrieval steps; teach
  observable state and concise rationale rather than depending on hidden model
  reasoning.
- Enrich self-reflection, iterative retrieval, answer synthesis, CRAG, and
  adaptive RAG with stopping rules, loop budgets, and failure paths.
- Split the oversized multi-agent notebook into network, supervisor, and
  hierarchical-team lessons.

Exit check: the learner can choose among a chain, deterministic graph, single
agent, and multi-agent system and defend the choice.

### Phase 6 — Production hardening and capstone

Deliverables:

- Separate conversational memory, retrieval memory, checkpoint persistence,
  prompt/KV caching, semantic answer caching, and traditional RAG.
- Add tracing, structured logs, per-stage latency, token/cost accounting, and
  quality monitoring.
- Add prompt-injection defenses, data exfiltration boundaries, PII handling,
  tenant isolation, and retrieval authorization.
- Add unit, integration, notebook smoke, and evaluation-regression tests.
- Add concurrency, timeout, retry, fallback, rate-limit, and cache-invalidation
  guidance.
- Build a capstone that uses the common dataset and publishes an architecture
  decision record plus final scorecard.

Exit check: the reference system is testable, observable, secure by design, and
has documented quality/cost/latency trade-offs.

## Decisions made now

| Decision | Reason |
|---|---|
| Preserve current notebooks until their replacements are verified | Avoid losing useful working code during restructuring |
| Use the documented target folder structure and migrate in verified batches | Navigation improves without a risky all-at-once rename |
| Use one shared dataset and golden set for core experiments | Comparisons become meaningful across techniques |
| Teach framework-free mechanics before framework convenience | The learner retains the concept when APIs change |
| Put evaluation before advanced agents in the learning order | Complex workflows need measurable value and stopping criteria |
| Treat notebooks as lessons and reusable modules as library code | Reduces copy/paste while keeping explanations close to experiments |
| Prefer official documentation and original papers as sources | Reduces version drift and secondary-source errors |

## Deferred decisions

These choices should be made during Phase 0 after the clean-run audit:

- Whether to rename all directories into a new sequential taxonomy or preserve
  paths and expose the sequence only through the curriculum index.
- Which hosted model provider is the default optional path. The core lessons
  should remain provider-swappable.
- Which evaluation library, if any, supplements the transparent in-repo metrics.
- Whether the capstone remains notebook-first or adds a small API/UI after the
  learning system is complete.

## Quality gate for each notebook

A notebook moves to `Complete` only when all are true:

- [ ] The 30-second summary and mental model are clear.
- [ ] The code runs top-to-bottom in the documented environment.
- [ ] The example uses repository-owned or reproducibly downloaded data.
- [ ] The technique is compared with an appropriate baseline.
- [ ] Expected output and at least one behavioral check are present.
- [ ] Trade-offs, failure modes, and production implications are explicit.
- [ ] Paid/network dependencies and estimated cost behavior are visible.
- [ ] Recall questions and one practice task are included.
- [ ] Sources are primary and version-sensitive claims are verified.
- [ ] Outputs are concise and contain no secrets or machine-specific paths.
- [ ] A second-pass technical and editorial review is complete.

## Working cadence

Work in small vertical slices:

1. Select one tracker item.
2. Capture its current behavior and failures.
3. Rebuild it using the canonical notebook contract.
4. Run and evaluate it against the shared dataset.
5. Review the rendered notebook for learning flow.
6. Update the tracker and any affected curriculum links.

The tracker is the operational source of truth; this document changes only when
the curriculum or quality bar changes.
