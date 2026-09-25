#!/usr/bin/env python3
"""Rebuild query expansion, decomposition, and HyDE lessons."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


COMMON_SETUP = r'''
import re

def tokens(text: str) -> list[str]:
    stop = {"a", "an", "and", "are", "does", "for", "how", "is", "of", "the", "to", "what", "who"}
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stop]

def overlap_rank(query: str, documents: list[dict]) -> list[dict]:
    query_terms = set(tokens(query))
    return sorted(
        documents,
        key=lambda document: (-len(query_terms & set(tokens(document["text"]))), document["id"]),
    )
'''


def build_expansion() -> None:
    build_standard_lesson(
        "10-query-enhancement/1-query_expansion.ipynb",
        {
            "stage": "Query transformation",
            "title": "Query Expansion: Add Recall Without Losing Intent",
            "difficulty": "Intermediate",
            "key_idea": "Expansion is a controlled vocabulary bridge. Every added term can recover a relevant document or introduce query drift.",
            "summary": "This notebook compares a lexical baseline with curated and aggressive expansion on a four-document policy corpus. Curated terms bridge a paraphrase to the incident policy; an unrelated payment expansion demonstrates drift.",
            "why": "Users rarely repeat a document's exact terminology. Expansion can bridge aliases and acronyms, but unconstrained generation can change the question and retrieve confidently irrelevant context.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Synonym/alias expansion, deduplication, drift checks, top-1 measurement | Live LLM expansion, multilingual thesaurus, neural retrieval |",
            "mental_model": "```text\noriginal query -> candidate terms -> deduplicate/limit -> retrieve -> compare with original intent\n```",
            "setup_code": COMMON_SETUP + r'''
documents = [
    {"id": "access", "text": "Role access requests require manager approval."},
    {"id": "billing", "text": "Invoice payment disputes receive a billing review."},
    {"id": "incidents", "text": "Priority one incidents are acknowledged within fifteen minutes."},
    {"id": "retention", "text": "Audit logs are retained for thirty days."},
]
query = "How quickly is a critical disruption recognized?"
expected_id = "incidents"
''',
            "how_it_works": "We normalize terms, add only approved aliases, and preserve the original query. Retrieval sees the union; evaluation still uses the original intent. Expansion size and source should be observable configuration.",
            "baseline_text": "The lexical baseline has no shared content term with the incident policy, so deterministic tie-breaking selects the wrong document.",
            "baseline_code": r'''
baseline_ranking = overlap_rank(query, documents)
[(item["id"], len(set(tokens(query)) & set(tokens(item["text"])))) for item in baseline_ranking]
''',
            "technique_text": "The curated expansion maps `critical disruption` to `priority one incident` and `recognized` to `acknowledged`. Terms are deduplicated and capped; the mapping is inspectable rather than generated silently.",
            "technique_code": r'''
CURATED_ALIASES = {
    "critical": ["priority", "one"],
    "disruption": ["incident"],
    "recognized": ["acknowledged"],
}

def expand(query: str, aliases: dict[str, list[str]], limit: int = 5) -> str:
    additions = []
    for term in tokens(query):
        additions.extend(aliases.get(term, []))
    additions = list(dict.fromkeys(additions))[:limit]
    return " ".join([query, *additions])

curated_query = expand(query, CURATED_ALIASES)
curated_query, overlap_rank(curated_query, documents)[0]["id"]
''',
            "experiment_text": "We compare the original, curated expansion, and an aggressive expansion that also adds unrelated billing terms. Hit@1 measures intent preservation on this single labeled query; expansion length is reported as a drift signal.",
            "experiment_code": r'''
aggressive_aliases = {**CURATED_ALIASES, "quickly": ["invoice", "payment", "billing"]}
variants = {
    "original": query,
    "curated": curated_query,
    "aggressive": expand(query, aggressive_aliases, limit=8),
}
results = {
    name: {
        "top_id": overlap_rank(value, documents)[0]["id"],
        "hit@1": float(overlap_rank(value, documents)[0]["id"] == expected_id),
        "added_terms": len(set(tokens(value)) - set(tokens(query))),
    }
    for name, value in variants.items()
}
results
''',
            "evaluation": "The original query misses. Curated expansion retrieves `incidents`; aggressive expansion drifts to `billing` because three unrelated added terms outweigh the intended bridge. This proves expansion needs caps, provenance, and regression labels.",
            "checks_code": r'''
assert results["original"]["hit@1"] == 0.0
assert results["curated"] == {"top_id": "incidents", "hit@1": 1.0, "added_terms": 4}
assert results["aggressive"]["top_id"] == "billing"
assert results["aggressive"]["added_terms"] > results["curated"]["added_terms"]
print("Query-expansion checks passed.")
''',
            "decision_guide": "| Need | Expansion source |\n|---|---|\n| Product aliases/acronyms | Governed domain dictionary |\n| Spelling variants | Deterministic normalization |\n| Broad discovery | Generated alternatives with caps and fusion |\n| High precision | Original query plus conservative aliases |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Irrelevant topic dominates | Expansion drift | Cap terms, weight original, validate intent |\n| Duplicate searches | Variants not normalized | Canonicalize and deduplicate |\n| No gain | Corpus already shares vocabulary | Skip expansion by segment |\n| Latency/cost spike | Too many variants | Parallelize with strict budget or fuse terms once |",
            "production_notes": "### Observability\nLog original/added terms, expansion source/version, variant count, branch hits, and drift rate.\n\n### Safety and Guardrails\nDo not let expansion add unauthorized entities or bypass metadata filters.\n\n### Latency and Cost\nCache deterministic expansions and cap generated variants before retrieval fan-out.",
            "practice": "Add one acronym and one ambiguous synonym. Define expected and forbidden document IDs before editing the alias map.",
            "recall": "Toggle - Recall: What is query drift?\nAdded terms change the original intent enough to favor irrelevant documents.\n\nToggle - Recall: Why keep the original query?\nIt anchors intent and provides a baseline/fusion branch.",
            "sources": "- [Query expansion survey](https://doi.org/10.1561/1500000010)\n- Repository-owned synthetic policy fixture in this notebook",
            "confidence": "High for the controlled drift example",
            "next_review": "Evaluate governed aliases on the shared golden set",
        },
    )


def build_decomposition() -> None:
    build_standard_lesson(
        "10-query-enhancement/2-query_decomposition.ipynb",
        {
            "stage": "Query transformation",
            "title": "Query Decomposition: Retrieve Each Required Fact",
            "difficulty": "Intermediate",
            "key_idea": "A multi-part answer is complete only when every atomic subquestion has evidence. Decomposition is an orchestration and coverage problem, not just prompt formatting.",
            "summary": "This notebook decomposes a two-part service question into owner and retention subqueries. A single top-1 retrieval covers one fact; parallel subqueries retrieve both sources and a deterministic synthesis cites each claim.",
            "why": "One embedding or lexical query can blur distinct intents. The highest-scoring document may answer only one clause while generation fills the rest from unsupported memory.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Atomic subqueries, parallel retrieval, evidence coverage, cited synthesis | Open-ended planning agent, live LLM, arbitrary dependency graphs |",
            "mental_model": "```text\ncompound question -> atomic questions -> retrieve each -> verify coverage -> synthesize with citations\n```",
            "setup_code": COMMON_SETUP + r'''
documents = [
    {"id": "service-owner", "text": "The Atlas service owner is Alice Chen."},
    {"id": "log-retention", "text": "Atlas audit logs are retained for thirty days."},
    {"id": "residency", "text": "Atlas production data is hosted in Mumbai."},
]
question = "Who owns Atlas and how long are its audit logs retained?"
required_sources = {"service-owner", "log-retention"}
''',
            "how_it_works": "We convert the compound question into independently answerable units, retrieve one source per unit, verify that the required evidence set is complete, and only then synthesize. Independent subqueries can run concurrently when they have no dependency.",
            "baseline_text": "A single top-1 search returns the log-retention document because it overlaps more terms, leaving the owner clause unsupported.",
            "baseline_code": r'''
baseline_hit = overlap_rank(question, documents)[0]
baseline_coverage = len({baseline_hit["id"]} & required_sources) / len(required_sources)
baseline_hit, baseline_coverage
''',
            "technique_text": "The decomposition is explicit and bounded. Each subquery has a named output slot, expected evidence type, and one retrieved document.",
            "technique_code": r'''
subqueries = {
    "owner": "Who is the Atlas service owner?",
    "retention": "How long are Atlas audit logs retained?",
}
evidence = {slot: overlap_rank(subquery, documents)[0] for slot, subquery in subqueries.items()}
{slot: item["id"] for slot, item in evidence.items()}
''',
            "experiment_text": "Completeness is the fraction of required sources retrieved. We compare the single-query top-1 baseline with the union of atomic top-1 results, then synthesize only from those records.",
            "experiment_code": r'''
retrieved_sources = {item["id"] for item in evidence.values()}
decomposed_coverage = len(retrieved_sources & required_sources) / len(required_sources)
answer = (
    "Alice Chen owns Atlas [service-owner]. "
    "Atlas audit logs are retained for thirty days [log-retention]."
)
results = {
    "baseline_coverage": baseline_coverage,
    "decomposed_coverage": decomposed_coverage,
    "retrieved_sources": sorted(retrieved_sources),
    "answer": answer,
}
results
''',
            "evaluation": "Single top-1 retrieval covers **1/2** required sources. Decomposition covers **2/2** and the synthesis cites the source for each claim. This tiny deterministic example demonstrates completeness, not automatic decomposition quality.",
            "checks_code": r'''
assert results["baseline_coverage"] == 0.5
assert results["decomposed_coverage"] == 1.0
assert set(results["retrieved_sources"]) == required_sources
assert all(f"[{source}]" in answer for source in required_sources)
print("Query-decomposition checks passed.")
''',
            "decision_guide": "| Question | Strategy |\n|---|---|\n| One fact/intent | Direct retrieval |\n| Independent clauses | Parallel decomposition |\n| Later step depends on earlier entity | Sequential plan |\n| Comparison | Retrieve same fields for each entity before synthesis |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Answer misses a clause | Subquery omitted | Coverage checklist against original question |\n| Duplicate work | Overlapping subqueries | Normalize/merge before retrieval |\n| Contradictory facts | Blind concatenation | Preserve provenance and reconciliation rule |\n| Cost explosion | Unbounded plan | Cap subqueries/depth and stop on coverage |",
            "production_notes": "### Observability\nTrace original question, subqueries, dependency graph, evidence IDs, per-step latency, coverage, and synthesis citations.\n\n### Safety and Guardrails\nEvery subquery inherits the caller's authorization scope.\n\n### Latency and Cost\nRun independent retrievals concurrently and enforce a total step/token budget.",
            "practice": "Add a dependent clause asking for the owner's department. Decide which step must run first and what evidence proves completion.",
            "recall": "Toggle - Recall: What makes a good subquery?\nIt is atomic, answerable, and traceable to a required output slot.\n\nToggle - Recall: Why measure coverage before synthesis?\nGeneration can sound complete even when evidence is missing.",
            "sources": "- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625)\n- Repository-owned synthetic service fixture in this notebook",
            "confidence": "High for the explicit two-clause example",
            "next_review": "Add dependent plans and conflicting evidence",
        },
    )


def build_hyde() -> None:
    build_standard_lesson(
        "10-query-enhancement/3-HyDE.ipynb",
        {
            "stage": "Query transformation",
            "title": "HyDE: Retrieve with a Hypothetical Answer Document",
            "difficulty": "Intermediate",
            "key_idea": "HyDE uses generated text as a retrieval representation, never as evidence. The final answer must come from retrieved sources.",
            "summary": "This notebook demonstrates HyDE with a transparent hypothetical document. A paraphrased urgency query has no lexical bridge to the incident policy; the hypothetical document supplies domain terms and retrieves it. A deliberately wrong duration shows why hypothetical facts cannot be cited.",
            "why": "Short questions may live far from document wording in embedding space. A plausible answer-shaped document can create a richer search representation, but it can also inject domain mismatch and hallucinated details.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Baseline vs hypothetical retrieval, evidence boundary, domain-mismatch check | Live LLM generation, neural embedding provider, production prompt benchmark |",
            "mental_model": "```text\nquestion -> hypothetical document -> retrieve real sources -> discard hypothesis -> answer from evidence\n```",
            "setup_code": COMMON_SETUP + r'''
documents = [
    {"id": "access", "text": "Manager approval is required before role access is granted."},
    {"id": "billing", "text": "Billing disputes receive an invoice review within five business days."},
    {"id": "incidents", "text": "Priority one incidents have a fifteen minute acknowledgement target."},
]
query = "How quickly does urgent downtime receive confirmation?"
expected_id = "incidents"
''',
            "how_it_works": "HyDE asks a generator for a plausible answer-like passage, embeds or searches with that passage, and retrieves real documents. The hypothetical document can improve vocabulary alignment but has zero evidentiary authority.",
            "baseline_text": "The lexical query shares no domain term with the incident policy, so deterministic tie-breaking returns the access document.",
            "baseline_code": r'''
baseline_top = overlap_rank(query, documents)[0]
baseline_top
''',
            "technique_text": "The hypothetical document maps urgency and confirmation into the corpus vocabulary. It intentionally invents a thirty-minute value so the evidence-boundary check is visible.",
            "technique_code": r'''
hypothetical_document = (
    "A priority one incident caused by downtime should receive an acknowledgement "
    "within thirty minutes."
)
hyde_top = overlap_rank(hypothetical_document, documents)[0]
hypothetical_document, hyde_top
''',
            "experiment_text": "We compare top-1 retrieval, then construct the final answer only from the retrieved policy. We also check whether the invented duration appears in the real evidence.",
            "experiment_code": r'''
baseline_hit = float(baseline_top["id"] == expected_id)
hyde_hit = float(hyde_top["id"] == expected_id)
invented_value_supported = "thirty" in hyde_top["text"].lower()
final_answer = "Priority one incidents have a fifteen minute acknowledgement target [incidents]."
results = {
    "baseline_hit@1": baseline_hit,
    "hyde_hit@1": hyde_hit,
    "hypothetical_duration_supported": invented_value_supported,
    "final_answer": final_answer,
}
results
''',
            "evaluation": "The original lexical query misses; the hypothetical document retrieves `incidents`. Its invented **thirty-minute** detail conflicts with the real **fifteen-minute** policy, so the final answer discards the hypothesis and cites only retrieved evidence.",
            "checks_code": r'''
assert results["baseline_hit@1"] == 0.0
assert results["hyde_hit@1"] == 1.0
assert not results["hypothetical_duration_supported"]
assert "fifteen" in results["final_answer"] and "thirty" not in results["final_answer"]
print("HyDE evidence-boundary checks passed.")
''',
            "decision_guide": "| Situation | Choice |\n|---|---|\n| Short abstract query, weak dense recall | Evaluate HyDE |\n| Exact IDs/names | Sparse/direct retrieval |\n| Strict latency/cost | Avoid extra generation unless gain is proven |\n| High hallucination consequence | Keep hypothesis isolated and require cited evidence |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Wrong domain retrieved | Hypothesis follows model prior | Domain prompt/router and baseline fusion |\n| Hypothetical fact appears in answer | Evidence boundary violated | Discard hypothesis before generation |\n| No gain | Query already matches corpus | Route around HyDE |\n| Latency doubles | Extra generation step | Cache, smaller model, timeout, measure by segment |",
            "production_notes": "### Observability\nLog hypothesis hash/text under approved policy, retrieved IDs, baseline-vs-HyDE delta, model/version, latency, and fallback.\n\n### Safety and Guardrails\nTreat hypothesis as untrusted generated text and never cite it.\n\n### Latency and Cost\nHyDE adds a generation call before retrieval; use only for segments with measured recall gain.",
            "practice": "Create a hypothesis for an out-of-domain query and verify that a minimum relevance threshold causes abstention rather than forced retrieval.",
            "recall": "Toggle - Recall: Is the hypothetical document evidence?\nNo. It is only a retrieval representation.\n\nToggle - Recall: What proves HyDE is useful?\nA controlled recall/quality gain that justifies the extra generation cost.",
            "sources": "- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496)\n- Repository-owned synthetic policy fixture in this notebook",
            "confidence": "High for the evidence-boundary demonstration",
            "next_review": "Evaluate real embeddings and domain-routed hypotheses",
        },
    )


if __name__ == "__main__":
    build_expansion()
    build_decomposition()
    build_hyde()
