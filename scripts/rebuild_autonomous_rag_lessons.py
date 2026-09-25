#!/usr/bin/env python3
"""Rebuild planning, reflection, iterative retrieval, and synthesis lessons."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_planning() -> None:
    build_standard_lesson(
        "section-16-autonomous-rag/3-COTRag.ipynb",
        {
            "stage": "Autonomous RAG patterns",
            "title": "Observable RAG Planning: Steps, Evidence, and Completion",
            "difficulty": "Advanced",
            "key_idea": "Planning should expose tasks and evidence requirements, not depend on hidden chain-of-thought.",
            "summary": "This notebook turns a two-part index-deletion question into an explicit plan: retrieve deletion policy, retrieve recovery policy, verify both sources, and synthesize. The plan is state a reviewer can inspect and test.",
            "why": "Multi-part questions need more than one source. Hidden reasoning cannot prove which requirements were covered or why the workflow stopped.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Explicit task plan, evidence slots, completion check, cited synthesis | Hidden reasoning capture, open-ended web planning, hosted LLM |",
            "mental_model": "```text\nquestion -> task list -> retrieve each evidence slot -> coverage gate -> cited answer\n```",
            "setup_code": r'''
DOCUMENTS = {
    "deletion": "Deleting an index requires owner approval and a recorded change ticket.",
    "recovery": "A deleted index can be rebuilt from the latest validated snapshot.",
}
question = "Can I delete an index, and how would I recover it?"
required_slots = {"deletion_policy", "recovery_policy"}
''',
            "how_it_works": "A planner emits named tasks with expected source IDs. Executors fill evidence slots; a coverage gate blocks synthesis until every required slot is present. The task list is concise operational state, not a transcript of private reasoning.",
            "baseline_text": "A single retrieval returns only the deletion policy, covering half the requested evidence.",
            "baseline_code": r'''
baseline_evidence = {"deletion_policy": {"source": "deletion", "text": DOCUMENTS["deletion"]}}
baseline_coverage = len(set(baseline_evidence) & required_slots) / len(required_slots)
baseline_coverage
''',
            "technique_text": "The plan has one retrieval task per evidence slot followed by verification and synthesis. Each task declares its output rather than an unbounded natural-language intention.",
            "technique_code": r'''
plan = [
    {"task": "retrieve", "slot": "deletion_policy", "source": "deletion"},
    {"task": "retrieve", "slot": "recovery_policy", "source": "recovery"},
    {"task": "verify_coverage", "requires": sorted(required_slots)},
    {"task": "synthesize", "requires": sorted(required_slots)},
]

evidence = {
    task["slot"]: {"source": task["source"], "text": DOCUMENTS[task["source"]]}
    for task in plan if task["task"] == "retrieve"
}
plan, evidence
''',
            "experiment_text": "We compare evidence coverage, assert that synthesis dependencies are explicit, and create an answer only after the coverage gate passes.",
            "experiment_code": r'''
coverage = len(set(evidence) & required_slots) / len(required_slots)
can_synthesize = required_slots <= set(evidence)
answer = None
if can_synthesize:
    answer = (
        f"{evidence['deletion_policy']['text']} [deletion] "
        f"{evidence['recovery_policy']['text']} [recovery]"
    )
results = {"baseline_coverage": baseline_coverage, "planned_coverage": coverage, "can_synthesize": can_synthesize, "answer": answer}
results
''',
            "evaluation": "The one-shot baseline covers **1/2** evidence slots. The explicit plan covers **2/2**, declares synthesis dependencies, and cites both sources. This demonstrates task completeness, not general planner intelligence.",
            "checks_code": r'''
assert results["baseline_coverage"] == 0.5 and results["planned_coverage"] == 1.0
assert results["can_synthesize"] and all(f"[{source}]" in answer for source in DOCUMENTS)
assert plan[-1]["requires"] == sorted(required_slots)
print("Observable planning checks passed.")
''',
            "decision_guide": "| Question shape | Approach |\n|---|---|\n| Single fact | Direct RAG |\n| Known multi-part requirements | Explicit task plan |\n| Dependent steps | DAG/sequential workflow |\n| Missing required evidence | Stop/abstain |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Missing clause | Plan omitted slot | Compare plan to question requirements |\n| Synthesis starts early | No coverage gate | Declare dependencies |\n| Plan grows forever | No task budget | Cap tasks/depth |\n| Plan text leaks reasoning | Free-form scratchpad stored | Keep concise task/action state |",
            "production_notes": "### Observability\nTrace task IDs, dependencies, evidence slots, source IDs, status, latency, and terminal reason.\n\n### Safety and Guardrails\nPlans cannot expand authorization or invent new tools.\n\n### Latency and Cost\nParallelize independent tasks and cap total tasks/retries.",
            "practice": "Add an approval-owner clause and update the required slots before adding retrieval code.",
            "recall": "Toggle - Recall: What makes a plan observable?\nNamed tasks, dependencies, evidence slots, and terminal conditions.\n\nToggle - Recall: Why avoid hidden reasoning logs?\nOperational state is enough to test control flow without collecting private rationale.",
            "sources": "- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for the explicit coverage contract",
            "next_review": "Add task failures and dynamic dependency validation",
        },
    )


def build_reflection() -> None:
    build_standard_lesson(
        "section-16-autonomous-rag/4-Selfreflection.ipynb",
        {
            "stage": "Autonomous RAG patterns",
            "title": "Self-Reflection: Rubric-Based, Evidence-Bound Revision",
            "difficulty": "Advanced",
            "key_idea": "Reflection needs an explicit rubric, inspectable evidence, and a revision limit. A second generation is not automatically a better answer.",
            "summary": "This notebook critiques an intentionally wrong retention answer against a source-backed rubric, performs one revision, and verifies factual value and citation. The loop cannot revise more than once.",
            "why": "Open-ended self-critique can rubber-stamp errors or loop indefinitely. Deterministic checks make the lesson's acceptance criteria visible.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Correctness/citation rubric, bounded revision, acceptance gate | LLM-as-judge calibration, style optimization, unlimited debate |",
            "mental_model": "```text\nevidence -> draft -> rubric critique -> revise once -> verify -> accept/abstain\n```",
            "setup_code": r'''
evidence = {"source": "retention", "text": "Audit logs are retained for thirty days."}
question = "How long are audit logs retained?"
MAX_REVISIONS = 1

def draft_answer() -> str:
    return "Audit logs are retained for ninety days [retention]."
''',
            "how_it_works": "The critic checks whether the evidence-backed value and source citation appear. Revision is allowed once and can only use the supplied evidence. Acceptance reruns the same rubric.",
            "baseline_text": "The initial draft is fluent and cited but contradicts the source, showing that citation presence alone is not faithfulness.",
            "baseline_code": r'''
draft = draft_answer()
draft
''',
            "technique_text": "The rubric returns structured booleans and issue labels. The reviser copies the supported claim rather than adding new facts.",
            "technique_code": r'''
def critique(answer: str) -> dict:
    lower = answer.lower()
    return {
        "supported_value": "thirty" in lower and "ninety" not in lower,
        "citation_present": "[retention]" in lower,
        "issues": [
            issue for issue, failed in (
                ("unsupported retention value", not ("thirty" in lower and "ninety" not in lower)),
                ("missing citation", "[retention]" not in lower),
            ) if failed
        ],
    }

initial_critique = critique(draft)
revised = f"{evidence['text']} [{evidence['source']}]" if initial_critique["issues"] else draft
final_critique = critique(revised)
initial_critique, revised, final_critique
''',
            "experiment_text": "We compare rubric pass rate before and after exactly one revision and require the final answer to contain no unsupported duration.",
            "experiment_code": r'''
def pass_rate(report: dict) -> float:
    return sum((report["supported_value"], report["citation_present"])) / 2

results = {
    "initial_pass_rate": pass_rate(initial_critique),
    "final_pass_rate": pass_rate(final_critique),
    "revisions": int(revised != draft),
    "final_answer": revised,
}
results
''',
            "evaluation": "The initial answer passes citation presence but fails factual support (**0.5 rubric pass rate**). One evidence-bound revision reaches **1.0** and removes `ninety`. This rule-based judge is reliable only for the explicit fixture.",
            "checks_code": r'''
assert results["initial_pass_rate"] == 0.5 and results["final_pass_rate"] == 1.0
assert results["revisions"] == MAX_REVISIONS
assert "thirty" in revised.lower() and "ninety" not in revised.lower()
assert final_critique["issues"] == []
print("Self-reflection checks passed.")
''',
            "decision_guide": "| Need | Mechanism |\n|---|---|\n| Exact fact/citation | Deterministic evidence check |\n| Nuanced quality | Calibrated judge plus human sample |\n| Missing evidence | Retrieve/abstain, not revise prose |\n| Repeated failure | Stop at revision budget |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Critic approves wrong answer | Weak rubric/shared bias | Grounded deterministic checks/human audit |\n| Revision invents facts | Evidence not constrained | Evidence-only revision contract |\n| Endless revisions | No budget | Hard iteration limit |\n| Score improves by gaming wording | Proxy overfit | Diverse labels and blind review |",
            "production_notes": "### Observability\nLog rubric version, issue labels, revision count, evidence IDs, acceptance, and latency/cost.\n\n### Safety and Guardrails\nCritique never grants permission to use new data or tools.\n\n### Latency and Cost\nRun cheap deterministic checks first and reserve model judges for unresolved cases.",
            "practice": "Add a completeness requirement and an answer that is correct but omits the citation.",
            "recall": "Toggle - Recall: What bounds reflection?\nA rubric, evidence set, and revision limit.\n\nToggle - Recall: Why can a judge be wrong?\nIt may share model bias, miss domain rules, or reward surface form.",
            "sources": "- [Self-Refine](https://arxiv.org/abs/2303.17651)\n- Repository-owned synthetic retention evidence",
            "confidence": "High for the deterministic rubric fixture",
            "next_review": "Calibrate model-based judges against human labels",
        },
    )


def build_query_plan() -> None:
    build_standard_lesson(
        "section-16-autonomous-rag/5-QueryPlanningdecomposition.ipynb",
        {
            "stage": "Autonomous RAG patterns",
            "title": "Query Planning: Parallel Retrieval and Dependent Comparison",
            "difficulty": "Advanced",
            "key_idea": "A query plan is a dependency graph: independent retrievals may run in parallel, but comparison waits for both typed results.",
            "summary": "This notebook plans a retention comparison between Atlas and Beacon. Two retrieval tasks are independent; a comparison task depends on both. Coverage and dependency checks prevent premature synthesis.",
            "why": "Basic decomposition lists questions. Planning also states execution order, output schema, and dependencies needed for derived conclusions.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| DAG dependencies, parallel-ready tasks, typed results, comparison | Async runtime benchmark, dynamic LLM planner, large workflow engine |",
            "mental_model": "```text\nretrieve Atlas --\\\n                  compare longer -> synthesize\nretrieve Beacon -/\n```",
            "setup_code": r'''
SOURCES = {
    "atlas": {"service": "Atlas", "retention_days": 30, "source": "atlas-policy"},
    "beacon": {"service": "Beacon", "retention_days": 90, "source": "beacon-policy"},
}
question = "Compare Atlas and Beacon log retention and identify which is longer."
''',
            "how_it_works": "Each node declares dependencies and an output key. The two retrieval tasks have none and can run concurrently; comparison depends on both outputs and emits a derived result with source provenance.",
            "baseline_text": "A single top-1 lookup returns Atlas only, so no valid comparison is possible.",
            "baseline_code": r'''
baseline = {"atlas": SOURCES["atlas"]}
baseline_can_compare = {"atlas", "beacon"} <= set(baseline)
baseline_can_compare
''',
            "technique_text": "The plan is a small DAG represented as data. A topological check ensures dependencies are satisfied before execution.",
            "technique_code": r'''
plan = [
    {"id": "get_atlas", "depends_on": [], "output": "atlas"},
    {"id": "get_beacon", "depends_on": [], "output": "beacon"},
    {"id": "compare", "depends_on": ["atlas", "beacon"], "output": "comparison"},
]
results = {}
for task in plan:
    assert set(task["depends_on"]) <= set(results), f"unsatisfied dependencies for {task['id']}"
    if task["id"] == "get_atlas": results["atlas"] = SOURCES["atlas"]
    elif task["id"] == "get_beacon": results["beacon"] = SOURCES["beacon"]
    else:
        longer = max((results["atlas"], results["beacon"]), key=lambda item: item["retention_days"])
        results["comparison"] = {"longer_service": longer["service"], "days": longer["retention_days"]}
results
''',
            "experiment_text": "We compare source coverage and verify that the derived comparison uses both typed values and retains both citations.",
            "experiment_code": r'''
citations = sorted((results["atlas"]["source"], results["beacon"]["source"]))
answer = (
    f"Atlas retains logs for {results['atlas']['retention_days']} days [atlas-policy]; "
    f"Beacon retains them for {results['beacon']['retention_days']} days [beacon-policy]. "
    f"{results['comparison']['longer_service']} is longer."
)
experiment = {"baseline_can_compare": baseline_can_compare, "planned_can_compare": "comparison" in results, "citations": citations, "answer": answer}
experiment
''',
            "evaluation": "The baseline cannot compare. The DAG retrieves both sources, computes `Beacon` as longer at **90 days**, and cites both policies. The plan demonstrates dependencies; it does not benchmark actual parallel speedup.",
            "checks_code": r'''
assert not experiment["baseline_can_compare"] and experiment["planned_can_compare"]
assert results["comparison"] == {"longer_service": "Beacon", "days": 90}
assert experiment["citations"] == ["atlas-policy", "beacon-policy"]
assert all(f"[{source}]" in answer for source in experiment["citations"])
print("Query-plan dependency checks passed.")
''',
            "decision_guide": "| Relationship | Execution |\n|---|---|\n| Independent facts | Parallel |\n| Entity discovered first | Sequential |\n| Comparison | Retrieve same schema for every entity |\n| Missing dependency | Stop/partial result with caveat |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Comparison uses one side | Dependency omitted | Typed dependency check |\n| Apples-to-oranges values | Schemas/units differ | Normalize before compare |\n| Slow despite independence | Serial executor | Parallel task group |\n| Partial result called complete | No coverage gate | Required outputs checklist |",
            "production_notes": "### Observability\nTrace DAG, task status, dependency waits, source IDs, units, latency, and partial failures.\n\n### Safety and Guardrails\nAll parallel branches inherit authorization; do not merge cross-tenant evidence.\n\n### Latency and Cost\nBound fan-out and parallelize only independent tasks.",
            "practice": "Add a third service with retention reported in months and insert a normalization dependency before comparison.",
            "recall": "Toggle - Recall: What distinguishes planning from a subquery list?\nDependencies, output schemas, and execution order.\n\nToggle - Recall: When is parallelism safe?\nWhen tasks do not depend on each other's outputs.",
            "sources": "- [LangGraph orchestrator-worker workflow](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic policy data",
            "confidence": "High for the explicit DAG fixture",
            "next_review": "Add async execution and partial-failure policy",
        },
    )


def build_iterative() -> None:
    build_standard_lesson(
        "section-16-autonomous-rag/6-Iterativeretrieval.ipynb",
        {
            "stage": "Autonomous RAG patterns",
            "title": "Iterative Retrieval: Accumulate Evidence, Stop Predictably",
            "difficulty": "Advanced",
            "key_idea": "Iterative retrieval needs a support threshold, query history, duplicate guard, and hard attempt budget.",
            "summary": "This notebook starts with `holiday allowance`, records a zero-support attempt, rewrites to `paid leave`, and succeeds on attempt two. An unsupported query exhausts the same two-attempt budget and abstains.",
            "why": "Repeating retrieval can bridge terminology, but without history and stopping rules it can issue duplicate searches or loop on an absent answer.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Query history, rewrite, evidence accumulation, duplicate guard, max attempts | Live model rewrite, web fallback, learned support grader |",
            "mental_model": "```text\nretrieve -> score sufficient? yes -> answer\n   | no -> new nonduplicate query (attempts left?) -> retrieve | else abstain\n```",
            "setup_code": r'''
import re

DOCUMENTS = [
    {"id": "access", "text": "Role access requires manager approval."},
    {"id": "leave", "text": "Employees receive twenty days of paid leave each year."},
]
MAX_ATTEMPTS = 2

def terms(text: str) -> set[str]: return set(re.findall(r"[a-z0-9]+", text.lower())) - {"a", "an", "do", "how", "is", "of", "the", "to"}

def retrieve(query: str) -> tuple[dict, int]:
    return max(((doc, len(terms(query) & terms(doc["text"]))) for doc in DOCUMENTS), key=lambda item: (item[1], item[0]["id"]))
''',
            "how_it_works": "Each attempt appends query, top source, and score. A zero score may trigger a governed rewrite only if the new query is not already in history and the budget remains.",
            "baseline_text": "One-shot retrieval for `holiday allowance` has zero support and returns a tie-broken irrelevant document.",
            "baseline_code": r'''
baseline_document, baseline_score = retrieve("holiday allowance")
baseline_document, baseline_score
''',
            "technique_text": "The loop accumulates trace records and returns a terminal reason. The rewrite map is explicit so the behavior is deterministic and testable.",
            "technique_code": r'''
REWRITES = {"holiday allowance": "paid leave", "weather tomorrow": "weather forecast"}

def iterative_retrieve(initial_query: str) -> dict:
    query, history = initial_query, []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if query in [item["query"] for item in history]:
            return {"terminal_reason": "duplicate_query", "history": history, "document": None}
        document, score = retrieve(query)
        history.append({"attempt": attempt, "query": query, "document_id": document["id"], "score": score})
        if score > 0:
            return {"terminal_reason": "supported", "history": history, "document": document}
        query = REWRITES.get(query, query + " policy")
    return {"terminal_reason": "budget_exhausted", "history": history, "document": None}

supported = iterative_retrieve("holiday allowance")
supported
''',
            "experiment_text": "We compare a resolvable vocabulary mismatch with an unsupported weather query, checking attempts, query uniqueness, and terminal reasons.",
            "experiment_code": r'''
unsupported = iterative_retrieve("weather tomorrow")
results = {"supported": supported, "unsupported": unsupported}
results
''',
            "evaluation": "The holiday query succeeds on attempt **2** with `leave`. The weather query makes two unique attempts and terminates `budget_exhausted` without an answer. No path can exceed two retrievals.",
            "checks_code": r'''
assert supported["terminal_reason"] == "supported" and supported["document"]["id"] == "leave"
assert len(supported["history"]) == 2 and supported["history"][1]["query"] == "paid leave"
assert unsupported["terminal_reason"] == "budget_exhausted" and len(unsupported["history"]) == MAX_ATTEMPTS
assert all(len({item["query"] for item in run["history"]}) == len(run["history"]) for run in results.values())
print("Iterative retrieval checks passed.")
''',
            "decision_guide": "| Result | Next action |\n|---|---|\n| Strong support | Stop and answer |\n| Vocabulary mismatch | One bounded rewrite |\n| Duplicate query | Stop |\n| Budget exhausted | Abstain/fallback with disclosure |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Same query repeats | No history guard | Deduplicate query fingerprints |\n| Loop never ends | Missing budget | Max attempts/time/cost |\n| Weak doc accepted | Threshold too low | Calibrate support labels |\n| Evidence conflicts | Accumulation without reconciliation | Provenance-aware synthesis |",
            "production_notes": "### Observability\nLog attempt, query hash/text per policy, ranked IDs/scores, accumulated evidence, and terminal reason.\n\n### Safety and Guardrails\nRewrites cannot broaden authorization or silently add web sources.\n\n### Latency and Cost\nBudget worst-case attempts and stop early on sufficient support.",
            "practice": "Add a rewrite that returns its original query and verify the duplicate guard stops before another retrieval.",
            "recall": "Toggle - Recall: What prevents infinite retrieval?\nAttempt budget plus duplicate-query detection.\n\nToggle - Recall: When should the loop stop early?\nAs soon as evidence meets the support threshold.",
            "sources": "- [LangGraph loops and branches](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic policy data",
            "confidence": "High for bounded deterministic iteration",
            "next_review": "Calibrate support thresholds on labeled failures",
        },
    )


def build_synthesis() -> None:
    build_standard_lesson(
        "section-16-autonomous-rag/7-answersynthesis.ipynb",
        {
            "stage": "Autonomous RAG patterns",
            "title": "Answer Synthesis: Reconcile Conflicts with Provenance",
            "difficulty": "Advanced",
            "key_idea": "Synthesis is not concatenation. Conflicting evidence needs scope, effective dates, and a declared resolution rule.",
            "summary": "This notebook merges two retention policies with different effective dates. For an as-of date after the migration, the newer 45-day policy controls; both sources remain visible and the conflict is disclosed.",
            "why": "Parallel retrieval can return contradictory facts. A fluent merger that hides disagreement creates false certainty and breaks citations.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Provenance records, effective-date rule, conflict detection, cited synthesis | Legal interpretation, semantic contradiction model, arbitrary source authority |",
            "mental_model": "```text\nevidence records -> normalize field/scope/date -> detect conflict -> resolve by rule -> answer + citations + caveat\n```",
            "setup_code": r'''
from datetime import date

evidence = [
    {"source": "policy-v1", "field": "retention_days", "value": 30, "effective": date(2025, 1, 1)},
    {"source": "migration-v2", "field": "retention_days", "value": 45, "effective": date(2026, 10, 1)},
]
as_of = date(2026, 11, 1)
''',
            "how_it_works": "Records share a field and scope, so differing values are a conflict. The declared rule selects the latest policy effective on or before `as_of`; the answer cites the controlling source and names the superseded value.",
            "baseline_text": "Naive concatenation reports both values without explaining which applies.",
            "baseline_code": r'''
baseline = "Logs are retained for 30 days [policy-v1] and 45 days [migration-v2]."
baseline
''',
            "technique_text": "The resolver filters out future records, sorts by effective date, detects distinct values, and retains every source for audit.",
            "technique_code": r'''
applicable = sorted((item for item in evidence if item["effective"] <= as_of), key=lambda item: item["effective"])
controlling = applicable[-1]
conflict = len({item["value"] for item in applicable}) > 1
resolution = {
    "value": controlling["value"],
    "source": controlling["source"],
    "conflict": conflict,
    "superseded": [item for item in applicable[:-1] if item["value"] != controlling["value"]],
}
resolution
''',
            "experiment_text": "We synthesize for two as-of dates—before and after migration—to verify the boundary and preserve citations.",
            "experiment_code": r'''
def resolve(as_of_date: date) -> dict:
    candidates = sorted((item for item in evidence if item["effective"] <= as_of_date), key=lambda item: item["effective"])
    if not candidates: return {"answer": None, "source": None}
    chosen = candidates[-1]
    return {
        "answer": f"As of {as_of_date.isoformat()}, retention is {chosen['value']} days [{chosen['source']}].",
        "source": chosen["source"],
    }

before = resolve(date(2026, 9, 30))
after = resolve(as_of)
results = {"before": before, "after": after, "conflict_detected": conflict}
results
''',
            "evaluation": "Before 2026-10-01, the 30-day policy controls. After it, the 45-day migration policy controls. The conflict is detected rather than averaged or hidden, and each answer cites its controlling source.",
            "checks_code": r'''
assert results["before"]["source"] == "policy-v1" and "30 days" in results["before"]["answer"]
assert results["after"]["source"] == "migration-v2" and "45 days" in results["after"]["answer"]
assert results["conflict_detected"] and resolution["superseded"][0]["source"] == "policy-v1"
print("Answer-synthesis conflict checks passed.")
''',
            "decision_guide": "| Evidence state | Synthesis |\n|---|---|\n| Consistent facts | Merge with citations |\n| Versioned conflict | Apply declared effective-date/authority rule |\n| Unresolved authority | Surface disagreement |\n| Missing required fact | Partial answer or abstain |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Two values silently averaged | Numeric merge without semantics | Field-specific conflict rule |\n| Old policy cited as current | Dates ignored | As-of filtering |\n| Caveat lost | Provenance stripped | Structured evidence through synthesis |\n| New source always wins | Recency mistaken for authority | Source ownership hierarchy |",
            "production_notes": "### Observability\nLog evidence IDs, normalized fields/units, conflicts, resolution rule/version, as-of date, and citations.\n\n### Safety and Guardrails\nNever merge evidence across unauthorized tenants or scopes.\n\n### Latency and Cost\nResolve deterministic metadata conflicts before invoking a model.",
            "practice": "Add a future-dated 60-day policy and prove it is ignored before its effective date.",
            "recall": "Toggle - Recall: What makes synthesis trustworthy?\nStructured provenance, explicit conflict rules, and citations.\n\nToggle - Recall: Should conflicting values be averaged?\nOnly if the metric definition explicitly supports aggregation; policy values generally should not be.",
            "sources": "- [LangGraph parallelization and synthesis](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic versioned policies",
            "confidence": "High for the effective-date rule fixture",
            "next_review": "Add authority hierarchies and unresolved conflicts",
        },
    )


if __name__ == "__main__":
    build_planning()
    build_reflection()
    build_query_plan()
    build_iterative()
    build_synthesis()
