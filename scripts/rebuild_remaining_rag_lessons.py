#!/usr/bin/env python3
"""Rebuild the multi-agent, corrective, adaptive, memory, and cache lessons."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_multiagent_overview() -> None:
    build_standard_lesson(
        "section-17-multi-agents-rags/8-multiagent.ipynb",
        {
            "stage": "Multi-agent RAG",
            "title": "Multi-Agent RAG: Choose the Smallest Coordination Topology",
            "difficulty": "Advanced",
            "key_idea": "More agents are justified only when specialization or ownership boundaries outweigh coordination cost.",
            "summary": "This compatibility overview compares network, supervisor, and hierarchical-team topologies on the same two-part policy task. Three focused companion notebooks implement each topology in isolation.",
            "why": "A monolithic multi-agent demo hides who owns routing, evidence, verification, and stopping. A topology decision should be explicit and testable.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Topology selection, ownership, message budget, companion lessons | Live model collaboration, distributed runtime, emergent agent behavior |",
            "mental_model": "```text\nnetwork: peers hand off\nsupervisor: one router delegates\nhierarchy: root -> team lead -> specialist\n```",
            "setup_code": r'''
task = {"needs": ["policy_lookup", "arithmetic"], "teams": 1, "stable_routing": True}
topologies = {
    "network": {"best_for": "peer expertise and flexible handoffs", "base_messages": 4},
    "supervisor": {"best_for": "stable centralized routing", "base_messages": 3},
    "hierarchy": {"best_for": "multiple teams and delegated ownership", "base_messages": 5},
}
''',
            "how_it_works": "Selection uses workload shape, not novelty. Network peers coordinate directly, a supervisor owns routing and synthesis, and a hierarchy adds team leads only when organizational boundaries need them.",
            "baseline_text": "Choosing a hierarchy for a single stable team adds two coordination messages without improving task coverage.",
            "baseline_code": r'''
baseline_choice = "hierarchy"
baseline = {"choice": baseline_choice, "coverage": 1.0, "messages": topologies[baseline_choice]["base_messages"]}
baseline
''',
            "technique_text": "A small decision rule picks a supervisor because routing is stable, the task has distinct skills, and only one team is involved.",
            "technique_code": r'''
def choose_topology(task: dict) -> str:
    if task["teams"] > 1:
        return "hierarchy"
    if task["stable_routing"]:
        return "supervisor"
    return "network"

choice = choose_topology(task)
decision = {"choice": choice, "coverage": 1.0, "messages": topologies[choice]["base_messages"]}
decision
''',
            "experiment_text": "We compare coverage and coordination messages while holding the task and successful specialist outputs constant.",
            "experiment_code": r'''
comparison = {
    "baseline": baseline,
    "selected": decision,
    "messages_saved": baseline["messages"] - decision["messages"],
    "companions": ["1-network.ipynb", "2-supervisor.ipynb", "3-hierarchical-teams.ipynb"],
}
comparison
''',
            "evaluation": "Both topologies cover the fixture, but the supervisor uses **3 messages instead of 5**. This is a coordination-cost comparison, not a claim that supervisors are universally best.",
            "checks_code": r'''
assert choice == "supervisor" and comparison["messages_saved"] == 2
assert comparison["baseline"]["coverage"] == comparison["selected"]["coverage"] == 1.0
assert len(comparison["companions"]) == 3
print("Multi-agent topology checks passed.")
''',
            "decision_guide": "| Condition | Topology |\n|---|---|\n| Flexible peer handoffs | Network |\n| Stable central routing | Supervisor |\n| Multiple teams/ownership layers | Hierarchy |\n| One capability is enough | Single agent |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Agent ping-pong | No owner/stop rule | Handoff budget and terminal owner |\n| Duplicate retrieval | Overlapping roles | Exclusive capability contracts |\n| Slow simple tasks | Over-orchestration | Single-agent fast path |\n| Untraceable answer | Provenance lost in messages | Typed evidence envelope |",
            "production_notes": "### Observability\nTrace topology, sender/receiver, role, evidence IDs, message count, and terminal owner.\n\n### Safety and Guardrails\nHandoffs cannot expand tool or data permissions.\n\n### Latency and Cost\nBudget messages and parallelize only independent specialist work.",
            "practice": "Change `teams` to two, explain the hierarchy decision, and assign a maximum message budget.",
            "recall": "Toggle - Recall: When is multi-agent RAG justified?\nWhen real specialization or ownership boundaries repay coordination cost.\n\nToggle - Recall: What should cross an agent boundary?\nTyped task state, evidence, provenance, and status—not an unbounded transcript.",
            "sources": "- [LangGraph multi-agent concepts](https://docs.langchain.com/oss/python/langchain/multi-agent)\n- Repository-owned synthetic topology fixture",
            "confidence": "High for the explicit topology rule",
            "next_review": "Benchmark real latency and failure recovery",
        },
    )


def build_network() -> None:
    build_standard_lesson(
        "section-17-multi-agents-rags/1-network.ipynb",
        {
            "stage": "Multi-agent RAG",
            "title": "Agent Network: Typed Peer Handoffs Without Ping-Pong",
            "difficulty": "Advanced",
            "key_idea": "A network needs typed handoffs, an owner, and a hop budget; peer freedom is not permission to loop.",
            "summary": "A policy specialist retrieves a retention fact, hands a typed calculation request to a math specialist, and receives a derived weekly value with provenance.",
            "why": "Peer-to-peer coordination fits fluid expertise, but unrestricted handoffs create loops, duplicated work, and ambiguous answer ownership.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Typed envelope, peer handoff, hop budget, provenance, terminal owner | Dynamic LLM routing, concurrent messaging, distributed transport |",
            "mental_model": "```text\npolicy peer --evidence+request--> math peer --derived result--> policy owner -> answer\n```",
            "setup_code": r'''
POLICY = {"source": "retention-policy", "days": 28}
request = {"question": "How many weeks are logs retained?", "owner": "policy", "hops": 0, "max_hops": 2}
''',
            "how_it_works": "The policy peer owns the user answer. It emits a handoff with a schema, evidence citation, and remaining budget. The math peer returns only the requested derived value.",
            "baseline_text": "An untyped message loses the source and gives the receiving peer no stopping contract.",
            "baseline_code": r'''
baseline_message = "Retention is 28 days. Convert it."
baseline_has_provenance = "source" in baseline_message.lower()
baseline_has_provenance
''',
            "technique_text": "The typed envelope retains provenance and increments the hop count exactly once per boundary crossing.",
            "technique_code": r'''
def policy_peer(state: dict) -> dict:
    assert state["hops"] < state["max_hops"]
    return {**state, "hops": state["hops"] + 1, "to": "math", "operation": "days_to_weeks", "evidence": POLICY}

def math_peer(message: dict) -> dict:
    assert message["operation"] == "days_to_weeks"
    return {"to": message["owner"], "weeks": message["evidence"]["days"] / 7, "source": message["evidence"]["source"], "hops": message["hops"] + 1}

handoff = policy_peer(request)
result = math_peer(handoff)
result
''',
            "experiment_text": "We require a correct derived value, preserved source, return to the original owner, and no hop-budget overflow.",
            "experiment_code": r'''
answer = f"Logs are retained for {result['weeks']:.0f} weeks [{result['source']}]."
experiment = {"answer": answer, "owner_restored": result["to"] == request["owner"], "within_budget": result["hops"] <= request["max_hops"]}
experiment
''',
            "evaluation": "The network completes in **2 hops**, returns to the policy owner, and preserves the source. The baseline exposes neither provenance nor a stopping rule.",
            "checks_code": r'''
assert not baseline_has_provenance
assert result == {"to": "policy", "weeks": 4.0, "source": "retention-policy", "hops": 2}
assert experiment["owner_restored"] and experiment["within_budget"]
print("Agent-network checks passed.")
''',
            "decision_guide": "| Need | Choice |\n|---|---|\n| Fluid peer expertise | Network |\n| Stable central routing | Supervisor |\n| Organizational subteams | Hierarchy |\n| No cross-skill dependency | Single agent |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Ping-pong | Peers can re-delegate forever | Hop budget and visited roles |\n| Lost citation | String-only handoff | Typed evidence envelope |\n| No final answer | No terminal owner | Preserve owner field |\n| Duplicate work | Roles overlap | Capability contracts |",
            "production_notes": "### Observability\nRecord handoff ID, roles, operation, evidence IDs, hop count, and terminal owner.\n\n### Safety and Guardrails\nEach peer receives least-privilege tools and data.\n\n### Latency and Cost\nUse direct calls for known dependencies; avoid a network for one skill.",
            "practice": "Add a currency-conversion peer and reject a third hop with a deterministic terminal reason.",
            "recall": "Toggle - Recall: What prevents ping-pong?\nA hop budget, visited-role guard, and terminal owner.\n\nToggle - Recall: Why type handoffs?\nTo preserve operation, evidence, provenance, and accountability.",
            "sources": "- [LangChain handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for the bounded handoff contract",
            "next_review": "Add peer failure and timeout handling",
        },
    )


def build_supervisor() -> None:
    build_standard_lesson(
        "section-17-multi-agents-rags/2-supervisor.ipynb",
        {
            "stage": "Multi-agent RAG",
            "title": "Supervisor Pattern: Central Routing and Evidence Ownership",
            "difficulty": "Advanced",
            "key_idea": "A supervisor owns routing and synthesis; specialists return bounded typed results rather than talking to the user.",
            "summary": "A deterministic supervisor routes policy and arithmetic questions to dedicated specialists, checks their result schema, and abstains on unsupported intents.",
            "why": "Central routing makes ownership and budgets obvious when intents are stable, but the supervisor becomes a bottleneck if it performs specialist work itself.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Intent routing, specialist contracts, synthesis ownership, abstention | Learned router, parallel runtime, model-generated tool calls |",
            "mental_model": "```text\nuser -> supervisor -> one specialist -> typed result -> supervisor -> answer\n```",
            "setup_code": r'''
POLICY = {"retention": {"value": 30, "unit": "days", "source": "policy-30"}}
queries = ["What is log retention?", "What is 6 times 7?", "What is tomorrow's weather?"]
''',
            "how_it_works": "The supervisor classifies each query, invokes one least-privilege specialist, validates the result type, and formats the final answer. Unsupported questions terminate without a handoff.",
            "baseline_text": "A route-everything-to-policy baseline answers only one of three intents correctly and risks hallucinating the others.",
            "baseline_code": r'''
baseline_routes = ["policy" for _ in queries]
baseline_correct = sum(route == expected for route, expected in zip(baseline_routes, ["policy", "math", "abstain"]))
baseline_correct
''',
            "technique_text": "Routing is deliberately transparent so branch behavior and errors can be labeled and tested.",
            "technique_code": r'''
def route(query: str) -> str:
    lower = query.lower()
    if "retention" in lower: return "policy"
    if "times" in lower: return "math"
    return "abstain"

def dispatch(query: str) -> dict:
    branch = route(query)
    if branch == "policy": return {"branch": branch, **POLICY["retention"]}
    if branch == "math": return {"branch": branch, "value": 6 * 7, "unit": None, "source": "calculator"}
    return {"branch": branch, "reason": "unsupported intent"}

results = [dispatch(query) for query in queries]
results
''',
            "experiment_text": "We score route accuracy and ensure only the supervisor formats user-facing answers.",
            "experiment_code": r'''
expected = ["policy", "math", "abstain"]
route_accuracy = sum(result["branch"] == label for result, label in zip(results, expected)) / len(expected)
answers = [
    (f"{item['value']} {item['unit'] or ''} [{item['source']}]".strip() if item["branch"] != "abstain" else "I cannot answer from the available specialists.")
    for item in results
]
{"route_accuracy": route_accuracy, "answers": answers}
''',
            "evaluation": "The supervisor routes **3/3** labeled queries correctly; the policy-only baseline routes **1/3**. The fixture tests orchestration, not natural-language intent coverage.",
            "checks_code": r'''
assert baseline_correct == 1 and route_accuracy == 1.0
assert results[0]["source"] == "policy-30" and results[1]["value"] == 42
assert results[2] == {"branch": "abstain", "reason": "unsupported intent"}
print("Supervisor checks passed.")
''',
            "decision_guide": "| Situation | Pattern |\n|---|---|\n| Stable mutually exclusive intents | Supervisor |\n| Flexible peer collaboration | Network |\n| Multiple managed teams | Hierarchy |\n| Router confidence is low | Clarify or abstain |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Wrong specialist | Weak labels | Route test set/confidence gate |\n| Supervisor does everything | Role leakage | Strict specialist contracts |\n| Bottleneck | Serial central work | Parallelize independent calls |\n| Hallucinated branch | No unsupported route | Explicit abstention |",
            "production_notes": "### Observability\nLog route label/confidence, specialist, schema validation, latency, evidence IDs, and terminal reason.\n\n### Safety and Guardrails\nThe supervisor cannot grant specialists extra permissions.\n\n### Latency and Cost\nUse a cheap router and bypass agents for deterministic operations.",
            "practice": "Add a retrieval-plus-calculation query and decide whether to invoke specialists sequentially or in parallel.",
            "recall": "Toggle - Recall: Who owns the final answer?\nThe supervisor.\n\nToggle - Recall: What is the supervisor's main risk?\nA routing or availability bottleneck that affects every branch.",
            "sources": "- [LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)\n- Repository-owned synthetic routing fixture",
            "confidence": "High for the labeled branch fixture",
            "next_review": "Add calibrated routing confidence",
        },
    )


def build_hierarchy() -> None:
    build_standard_lesson(
        "section-17-multi-agents-rags/3-hierarchical-teams.ipynb",
        {
            "stage": "Multi-agent RAG",
            "title": "Hierarchical Teams: Delegated Ownership with Bounded Depth",
            "difficulty": "Advanced",
            "key_idea": "A hierarchy should mirror real team boundaries and cap delegation depth; extra layers are not free reasoning.",
            "summary": "A root supervisor delegates a policy question to a policy-team lead, which invokes retrieval and verification specialists before returning a typed team result.",
            "why": "Hierarchies isolate tools and ownership across teams, but excessive layers amplify latency, context loss, and unclear failure responsibility.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Root/team roles, depth budget, retrieval-verification split, result contract | Dynamic organization design, distributed queues, live LLM teams |",
            "mental_model": "```text\nroot supervisor -> policy lead -> retriever -> verifier -> policy lead -> root\n```",
            "setup_code": r'''
DOCUMENT = {"source": "security-policy", "text": "Production access requires manager approval."}
request = {"intent": "policy", "question": "Who approves production access?", "depth": 0, "max_depth": 2}
''',
            "how_it_works": "The root chooses a team, the team lead owns specialist sequencing, and the root accepts only a verified team result. Depth counts delegation layers, not internal function calls.",
            "baseline_text": "A flat root-to-retriever path finds text but has no independent verification status or team ownership.",
            "baseline_code": r'''
baseline = {"evidence": DOCUMENT, "verified": False, "team": None}
baseline
''',
            "technique_text": "The root delegates one level; the team lead coordinates retrieval and verification within the final allowed depth.",
            "technique_code": r'''
def policy_team(state: dict) -> dict:
    assert state["depth"] <= state["max_depth"]
    evidence = DOCUMENT
    verified = "manager approval" in evidence["text"].lower()
    return {"team": "policy", "claim": "A manager approves production access.", "source": evidence["source"], "verified": verified, "depth": state["depth"]}

def root_supervisor(state: dict) -> dict:
    assert state["depth"] < state["max_depth"]
    delegated = {**state, "depth": state["depth"] + 1}
    return policy_team(delegated)

team_result = root_supervisor(request)
team_result
''',
            "experiment_text": "We compare verification and ownership while asserting the delegation depth stays within the declared limit.",
            "experiment_code": r'''
answer = f"{team_result['claim']} [{team_result['source']}]" if team_result["verified"] else "Insufficient verified evidence."
experiment = {"baseline_verified": baseline["verified"], "hierarchy_verified": team_result["verified"], "within_depth": team_result["depth"] <= request["max_depth"], "answer": answer}
experiment
''',
            "evaluation": "The hierarchy adds explicit policy-team ownership and verification within **depth 1 of 2**. It is warranted only if that boundary reflects real governance.",
            "checks_code": r'''
assert not experiment["baseline_verified"] and experiment["hierarchy_verified"]
assert experiment["within_depth"] and team_result["team"] == "policy"
assert "[security-policy]" in answer
print("Hierarchical-team checks passed.")
''',
            "decision_guide": "| Need | Pattern |\n|---|---|\n| Separate teams/tool domains | Hierarchy |\n| One stable dispatcher | Supervisor |\n| Peer handoffs | Network |\n| Single team and simple task | Avoid hierarchy |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Context loss | Too many summaries | Typed result schema/provenance |\n| Deep recursion | No depth budget | Hard max depth |\n| Slow answer | Serial layers | Remove unjustified layer |\n| Unclear failure owner | Delegation without status | Team-level terminal reason |",
            "production_notes": "### Observability\nTrace root/team task IDs, depth, specialist status, evidence IDs, verification, and failure owner.\n\n### Safety and Guardrails\nTool and tenant permissions remain scoped to each team.\n\n### Latency and Cost\nCharge a layer budget and execute independent specialists concurrently.",
            "practice": "Add a compliance team and make the root reject cross-team results without both team signatures.",
            "recall": "Toggle - Recall: When is a hierarchy useful?\nWhen real teams or permission domains need delegated ownership.\n\nToggle - Recall: What bounds it?\nA delegation-depth budget and typed team results.",
            "sources": "- [LangGraph hierarchical agent teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for the bounded delegation example",
            "next_review": "Add multi-team parallel aggregation",
        },
    )


def build_corrective_rag() -> None:
    build_standard_lesson(
        "section-18-corrective-rag/2-CorrectiveRAG.ipynb",
        {
            "stage": "Corrective and adaptive RAG",
            "title": "Corrective RAG: Grade, Rewrite, Fall Back, or Abstain",
            "difficulty": "Advanced",
            "key_idea": "Corrective RAG needs a calibrated relevance gate and bounded recovery policy; a grader score alone is not a correction.",
            "summary": "A weak local retrieval for a backup-encryption question is graded below threshold, rewritten once, and resolved from a bounded fallback collection with a citation.",
            "why": "Generation cannot repair missing evidence. Corrective RAG makes retrieval failure observable and chooses a controlled recovery branch before answering.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Transparent grader, threshold, one rewrite, bounded fallback, abstention | Web search, learned relevance model, production threshold calibration |",
            "mental_model": "```text\nretrieve -> grade >= threshold? answer : rewrite once -> fallback -> grade -> answer/abstain\n```",
            "setup_code": r'''
import re

LOCAL = [{"id": "local-access", "text": "Backup access requires manager approval."}]
FALLBACK = [{"id": "security-backup", "text": "Backups are encrypted with AES-256 at rest."}]
THRESHOLD = 0.5
MAX_REWRITES = 1

def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"are", "at", "is", "the", "with"}

def grade(query: str, document: dict) -> float:
    return len(terms(query) & terms(document["text"])) / max(1, len(terms(query)))
''',
            "how_it_works": "The controller records a retrieval score, applies an explicit threshold, rewrites at most once, and searches only the approved fallback collection. Evidence below threshold never reaches generation.",
            "baseline_text": "Local top-1 returns an access document that overlaps on `backup` but cannot support the requested encryption claim.",
            "baseline_code": r'''
question = "How are backups encrypted at rest?"
baseline_doc = LOCAL[0]
baseline_score = grade(question, baseline_doc)
{"document": baseline_doc["id"], "score": baseline_score, "answerable": baseline_score >= THRESHOLD}
''',
            "technique_text": "The recovery branch rewrites the query to the missing concept and grades the fallback result under the same rule.",
            "technique_code": r'''
def corrective_retrieve(query: str) -> dict:
    local = max(LOCAL, key=lambda doc: grade(query, doc))
    trace = [{"branch": "local", "query": query, "doc": local["id"], "score": grade(query, local)}]
    if trace[-1]["score"] >= THRESHOLD:
        return {"status": "answer", "evidence": local, "trace": trace, "rewrites": 0}
    rewritten = "backup encryption AES at rest"
    fallback = max(FALLBACK, key=lambda doc: grade(rewritten, doc))
    trace.append({"branch": "fallback", "query": rewritten, "doc": fallback["id"], "score": grade(rewritten, fallback)})
    status = "answer" if trace[-1]["score"] >= THRESHOLD else "abstain"
    return {"status": status, "evidence": fallback if status == "answer" else None, "trace": trace, "rewrites": 1}

corrected = corrective_retrieve(question)
corrected
''',
            "experiment_text": "We verify branch choice, score improvement, rewrite budget, and citation-gated answer generation.",
            "experiment_code": r'''
answer = (
    f"{corrected['evidence']['text']} [{corrected['evidence']['id']}]"
    if corrected["status"] == "answer" else "Insufficient evidence."
)
experiment = {"baseline_score": baseline_score, "final_score": corrected["trace"][-1]["score"], "rewrites": corrected["rewrites"], "answer": answer}
experiment
''',
            "evaluation": "The local score is below **0.5**; one rewrite reaches the approved fallback and passes at **0.75**, producing a cited answer. The lexical grader is intentionally narrow and must be calibrated before production use.",
            "checks_code": r'''
assert experiment["baseline_score"] < THRESHOLD <= experiment["final_score"]
assert experiment["rewrites"] == MAX_REWRITES and len(corrected["trace"]) == 2
assert corrected["status"] == "answer" and "[security-backup]" in answer
print("Corrective-RAG checks passed.")
''',
            "decision_guide": "| Grade/result | Action |\n|---|---|\n| Strong support | Answer with citations |\n| Weak query match | Rewrite once |\n| Approved alternate source exists | Bounded fallback |\n| Still weak/no source | Abstain |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Wrong evidence passes | Threshold/grader miscalibrated | Labeled calibration set |\n| Endless rewrites | No retry budget | Hard rewrite limit |\n| Unsafe web leakage | Unbounded fallback | Approved sources only |\n| Fluent unsupported answer | Generation before gate | Citation/support gate |",
            "production_notes": "### Observability\nLog scores, threshold version, query rewrites, branch, evidence IDs, retries, and terminal reason.\n\n### Safety and Guardrails\nFallback respects tenant, ACL, and source allowlists.\n\n### Latency and Cost\nRun correction only below threshold and cap alternate searches.",
            "practice": "Add an unanswerable salary query and assert that fallback ends in abstention after one rewrite.",
            "recall": "Toggle - Recall: What makes RAG corrective?\nA retrieval-quality gate plus a bounded recovery branch.\n\nToggle - Recall: When should it stop?\nAt sufficient evidence or the explicit retry/fallback budget.",
            "sources": "- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for the deterministic control-flow fixture",
            "next_review": "Calibrate thresholds on labeled retrieval data",
        },
    )


def build_adaptive_rag() -> None:
    build_standard_lesson(
        "section-19-adaptive-rag/adaptive-rag.ipynb",
        {
            "stage": "Corrective and adaptive RAG",
            "title": "Adaptive RAG: Route by Evidence Need, Not Prompt Length",
            "difficulty": "Advanced",
            "key_idea": "Adaptive RAG selects the cheapest branch that can satisfy the evidence requirement, with a shared step and cost budget.",
            "summary": "A transparent router sends a direct fact lookup to one retrieval, a comparison to two retrievals plus synthesis, and an unsupported weather request to abstention.",
            "why": "Running the most capable workflow for every query wastes latency and cost; running the simplest one for every query misses compound evidence needs.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Route labels, branch contracts, coverage, step/cost budget, abstention | Learned router, online model pricing, dynamic tool marketplace |",
            "mental_model": "```text\nclassify evidence need -> direct | compare | abstain -> enforce budget -> cited result\n```",
            "setup_code": r'''
POLICIES = {"Atlas": {"days": 30, "source": "atlas"}, "Beacon": {"days": 90, "source": "beacon"}}
cases = [
    ("How long does Atlas retain logs?", "direct"),
    ("Which retains logs longer, Atlas or Beacon?", "compare"),
    ("Will it rain tomorrow?", "abstain"),
]
MAX_STEPS = 3
MAX_COST = 3
''',
            "how_it_works": "The route is an evidence-shape label. Each branch declares expected steps and cost units; a controller rejects branches that exceed the shared budget before execution.",
            "baseline_text": "Always using the comparison branch retrieves two sources even for a one-fact question and still cannot answer weather.",
            "baseline_code": r'''
baseline_cost = [3, 3, 3]
baseline_total_cost = sum(baseline_cost)
baseline_total_cost
''',
            "technique_text": "The router distinguishes a single known entity, a cross-entity comparison, and unsupported scope.",
            "technique_code": r'''
def route(query: str) -> str:
    lower = query.lower()
    if "weather" in lower or "rain" in lower: return "abstain"
    if "which" in lower or "compare" in lower: return "compare"
    return "direct"

def run(query: str) -> dict:
    branch = route(query)
    if branch == "abstain": return {"branch": branch, "steps": 1, "cost": 0, "sources": []}
    if branch == "direct": return {"branch": branch, "steps": 2, "cost": 1, "sources": [POLICIES["Atlas"]["source"]], "answer": "30 days"}
    longer = max(POLICIES, key=lambda name: POLICIES[name]["days"])
    return {"branch": branch, "steps": 3, "cost": 3, "sources": ["atlas", "beacon"], "answer": f"{longer}: {POLICIES[longer]['days']} days"}

results = [run(query) for query, _ in cases]
results
''',
            "experiment_text": "We score route accuracy, assert every branch stays within budget, and compare total cost with the always-complex baseline.",
            "experiment_code": r'''
route_accuracy = sum(result["branch"] == expected for result, (_, expected) in zip(results, cases)) / len(cases)
adaptive_cost = sum(result["cost"] for result in results)
within_budget = all(result["steps"] <= MAX_STEPS and result["cost"] <= MAX_COST for result in results)
experiment = {"route_accuracy": route_accuracy, "adaptive_cost": adaptive_cost, "baseline_cost": baseline_total_cost, "within_budget": within_budget}
experiment
''',
            "evaluation": "The router labels **3/3** cases correctly and uses **4 cost units instead of 9**, while every branch stays within three steps. Savings depend on the chosen cost model and case mix.",
            "checks_code": r'''
assert experiment == {"route_accuracy": 1.0, "adaptive_cost": 4, "baseline_cost": 9, "within_budget": True}
assert results[1]["sources"] == ["atlas", "beacon"] and results[2]["branch"] == "abstain"
print("Adaptive-RAG checks passed.")
''',
            "decision_guide": "| Evidence need | Branch |\n|---|---|\n| One source/fact | Direct retrieval |\n| Multiple sources/derived result | Compound workflow |\n| Weak retrieval | Corrective branch |\n| Unsupported/out of scope | Abstain |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Expensive simple queries | Over-routing | Complexity labels/cost objective |\n| Missing comparison clause | Under-routing | Evidence-coverage tests |\n| Branch loop | No shared budget | Global step ceiling |\n| Silent route drift | No labels | Route confusion matrix |",
            "production_notes": "### Observability\nTrack route label/confidence, branch, steps, evidence coverage, latency, cost, and terminal reason.\n\n### Safety and Guardrails\nAll branches enforce the same ACL and citation policy.\n\n### Latency and Cost\nOptimize expected quality under explicit p95 and spend budgets.",
            "practice": "Add a weak-evidence case that routes to corrective RAG and remains within the global budget.",
            "recall": "Toggle - Recall: What should routing predict?\nThe evidence/workflow need, not merely query length.\n\nToggle - Recall: What bounds all branches?\nShared quality, safety, step, latency, and cost contracts.",
            "sources": "- [Adaptive-RAG](https://arxiv.org/abs/2403.14403)\n- Repository-owned synthetic routing fixture",
            "confidence": "High for the labeled routing fixture",
            "next_review": "Add a corrective branch and calibration curves",
        },
    )


def build_memory_rag() -> None:
    build_standard_lesson(
        "section-20-rag-with-persistant-memory/ragmemory.ipynb",
        {
            "stage": "Stateful RAG",
            "title": "RAG Memory: Separate Conversation, Checkpoint, and Long-Term State",
            "difficulty": "Advanced",
            "key_idea": "Memory is not one bucket: conversational context, workflow checkpoints, and long-term user data have different lifetimes and privacy contracts.",
            "summary": "A pronoun follow-up uses session memory, a workflow resumes from a checkpoint, and an opt-in preference expires from long-term memory at its TTL.",
            "why": "Mixing all state into the prompt leaks data, prevents reliable resume, and makes deletion or expiry impossible to reason about.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Three memory classes, stable keys, checkpoint resume, consent, TTL/expiry | Vector memory store, encryption implementation, cross-device sync |",
            "mental_model": "```text\nsession turns -> resolve current conversation\ncheckpoint -> resume workflow exactly\nlong-term profile -> opt-in fact with TTL/delete\n```",
            "setup_code": r'''
from datetime import date

session_id = "session-demo"
conversation = [{"role": "user", "text": "Tell me about Atlas retention."}, {"role": "assistant", "entity": "Atlas", "text": "Atlas retains logs for 30 days."}]
checkpoint = {"thread_id": "thread-demo", "next_step": "synthesize", "evidence_ids": ["atlas-policy"], "status": "paused"}
profile = {"user_id": "user-demo", "fact": "prefers concise answers", "consent": True, "expires_on": date(2026, 10, 1)}
TODAY = date(2026, 9, 25)
''',
            "how_it_works": "Session context resolves references only within the current conversation. Checkpoints store control state for deterministic resume. Long-term facts require consent, minimal scope, an expiry date, and a deletion path.",
            "baseline_text": "A single untyped memory list cannot distinguish what belongs in a prompt, what resumes execution, or what must expire.",
            "baseline_code": r'''
baseline_memory = conversation + [checkpoint, profile]
baseline_types_declared = all("memory_type" in item for item in baseline_memory)
baseline_types_declared
''',
            "technique_text": "Typed accessors expose only the memory class required for the current operation.",
            "technique_code": r'''
def resolve_followup(turns: list[dict], question: str) -> str:
    entity = next(turn["entity"] for turn in reversed(turns) if "entity" in turn)
    return question.replace("it", entity)

def resume(saved: dict) -> dict:
    assert saved["status"] == "paused"
    return {**saved, "status": "running", "resumed_at": saved["next_step"]}

def read_profile(record: dict, as_of: date) -> str | None:
    return record["fact"] if record["consent"] and as_of < record["expires_on"] else None

resolved = resolve_followup(conversation, "How long does it retain logs?")
resumed = resume(checkpoint)
preference_now = read_profile(profile, TODAY)
resolved, resumed, preference_now
''',
            "experiment_text": "We verify session resolution, exact checkpoint resume, long-term access before expiry, and automatic denial at the boundary.",
            "experiment_code": r'''
preference_at_expiry = read_profile(profile, profile["expires_on"])
deleted_profile = {**profile, "consent": False}
preference_after_delete = read_profile(deleted_profile, TODAY)
experiment = {"resolved": resolved, "resumed_at": resumed["resumed_at"], "available_now": preference_now, "at_expiry": preference_at_expiry, "after_delete": preference_after_delete}
experiment
''',
            "evaluation": "Session memory resolves `it` to `Atlas`; the checkpoint resumes at `synthesize`; the profile is readable before **2026-10-01** and unavailable at expiry or after consent deletion.",
            "checks_code": r'''
assert not baseline_types_declared
assert experiment["resolved"] == "How long does Atlas retain logs?"
assert experiment["resumed_at"] == "synthesize" and experiment["available_now"] == "prefers concise answers"
assert experiment["at_expiry"] is None and experiment["after_delete"] is None
print("RAG-memory checks passed.")
''',
            "decision_guide": "| State | Store/lifetime |\n|---|---|\n| Current dialogue reference | Session/conversation |\n| Workflow resume point | Checkpoint by thread |\n| Reusable user preference | Opt-in long-term with TTL |\n| Sensitive unnecessary data | Do not store |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Cross-user leak | Weak keys | Tenant/user/session isolation |\n| Wrong resume | Prompt summary used as checkpoint | Typed workflow state |\n| Stale preference | No TTL/version | Expiry and refresh |\n| Impossible deletion | No provenance/index | Stable record IDs and delete audit |",
            "production_notes": "### Observability\nLog memory type, key, read/write purpose, consent, TTL, version, deletion, and resume point—never raw secrets.\n\n### Safety and Guardrails\nMinimize stored data, isolate tenants, encrypt, authorize reads, and support export/delete.\n\n### Latency and Cost\nRetrieve only memory needed for the current step and summarize bounded session windows.",
            "practice": "Add a second session and prove its pronoun resolution cannot read the first session's entity.",
            "recall": "Toggle - Recall: Why are checkpoints not conversation memory?\nThey preserve executable workflow state, not just dialogue meaning.\n\nToggle - Recall: What must long-term memory have?\nPurpose, consent, identity, TTL/version, access control, and deletion.",
            "sources": "- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)\n- Repository-owned synthetic memory fixture",
            "confidence": "High for lifecycle separation and boundary checks",
            "next_review": "Add tenant isolation and encrypted-store integration",
        },
    )


def build_cache_rag() -> None:
    build_standard_lesson(
        "section-21-cache-rag/cache_augment_generation.ipynb",
        {
            "stage": "Performance and caching",
            "title": "Cache-Augmented Generation: Key Scope, Reuse, and Invalidation",
            "difficulty": "Advanced",
            "key_idea": "Prompt/context reuse and semantic answer caching solve different problems and need different correctness boundaries.",
            "summary": "A versioned context cache safely reuses a prepared policy prefix; a semantic answer cache demonstrates a stale date-scoped answer and is rejected by an as-of/version guard.",
            "why": "Caching can reduce latency and cost, but a fast stale answer is a correctness incident. Cache identity must include every input that changes meaning.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Context/prompt cache, answer-cache contrast, version/as-of keys, invalidation, hit metrics | Provider billing details, distributed eviction, embedding-based ANN cache |",
            "mental_model": "```text\nimmutable prepared context + version -> context cache\nquery meaning + policy/version/as-of -> answer cache (higher risk)\nsource change -> invalidate/re-key\n```",
            "setup_code": r'''
from hashlib import sha256

policies = {
    "v1": {"effective": "2025-01-01", "days": 30, "text": "Logs are retained for 30 days."},
    "v2": {"effective": "2026-10-01", "days": 45, "text": "Logs are retained for 45 days."},
}
context_cache = {}
answer_cache = {}

def digest(*parts: str) -> str:
    return sha256("|".join(parts).encode()).hexdigest()[:12]
''',
            "how_it_works": "The context cache keys prepared immutable source content by knowledge-base version and prompt template. Answer caching additionally needs query semantics, authorization, model/prompt version, and temporal scope.",
            "baseline_text": "A semantic answer cache keyed only by normalized wording returns the v1 answer even after the v2 policy becomes effective.",
            "baseline_code": r'''
semantic_key = "how long are logs retained"
answer_cache[semantic_key] = {"answer": "30 days", "policy_version": "v1"}
stale_hit = answer_cache[semantic_key]
stale_hit
''',
            "technique_text": "A safer context key incorporates source version and template. An answer guard requires the active policy version and as-of date to match before reuse.",
            "technique_code": r'''
def prepared_context(version: str, template_version: str = "prompt-v1") -> tuple[str, bool]:
    key = digest("context", version, template_version, policies[version]["text"])
    hit = key in context_cache
    context_cache.setdefault(key, policies[version]["text"])
    return context_cache[key], hit

def reusable_answer(entry: dict, active_version: str, as_of: str) -> bool:
    return entry["policy_version"] == active_version and entry.get("as_of") == as_of

context_first, first_hit = prepared_context("v2")
context_second, second_hit = prepared_context("v2")
guarded_reuse = reusable_answer(stale_hit, "v2", "2026-10-02")
first_hit, second_hit, guarded_reuse
''',
            "experiment_text": "We require a miss-then-hit for unchanged context and rejection of the stale answer after the policy boundary.",
            "experiment_code": r'''
fresh_answer = {"answer": "45 days", "policy_version": "v2", "as_of": "2026-10-02", "source": "policy-v2"}
final = stale_hit if guarded_reuse else fresh_answer
metrics = {"context_hits": int(first_hit) + int(second_hit), "context_requests": 2, "stale_answer_reused": guarded_reuse, "final": final}
metrics
''',
            "evaluation": "The prepared context produces a deterministic **miss then hit**. The incomplete semantic key is rejected, so the final answer is **45 days [policy-v2]** for the new effective period.",
            "checks_code": r'''
assert first_hit is False and second_hit is True and context_first == context_second
assert not metrics["stale_answer_reused"]
assert metrics["final"]["answer"] == "45 days" and metrics["final"]["source"] == "policy-v2"
print("Cache-RAG checks passed.")
''',
            "decision_guide": "| Reuse target | Risk/requirement |\n|---|---|\n| Tokenized/prefilled stable prefix | Versioned context key |\n| Exact deterministic tool result | Exact inputs and TTL |\n| Semantic final answer | High risk; scope/version/auth guards |\n| Volatile or personalized claim | Bypass or very short TTL |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Stale answer | Source version absent | Version key/event invalidation |\n| Cross-user leak | Auth scope absent | Tenant/user permissions in key |\n| Low hit rate | Over-specific unstable key | Separate stable context from query |\n| False semantic hit | Similar wording, different scope | As-of/entity/intent guards |",
            "production_notes": "### Observability\nTrack cache layer, key schema version, hit/miss, age, source version, invalidation reason, and correctness samples.\n\n### Safety and Guardrails\nInclude authorization scope; never share personalized answers across principals.\n\n### Latency and Cost\nMeasure end-to-end savings after lookup, serialization, and validation overhead.",
            "practice": "Add tenant ID and prompt version to the answer key, then test that either change forces a miss.",
            "recall": "Toggle - Recall: What is safer to cache?\nStable prepared context is generally safer than a final semantic answer.\n\nToggle - Recall: What invalidates cached RAG state?\nSource, prompt/model, authorization, temporal scope, or answer-policy changes.",
            "sources": "- [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for version and temporal invalidation behavior",
            "next_review": "Add tenant keys, TTL metrics, and concurrency tests",
        },
    )


if __name__ == "__main__":
    build_multiagent_overview()
    build_network()
    build_supervisor()
    build_hierarchy()
    build_corrective_rag()
    build_adaptive_rag()
    build_memory_rag()
    build_cache_rag()
