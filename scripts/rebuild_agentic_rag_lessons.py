#!/usr/bin/env python3
"""Rebuild the overlapping Agentic RAG and retrieval-tool ReAct lessons."""

from pathlib import Path
from shutil import copyfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_agentic_workflow() -> None:
    build_standard_lesson(
        "Section-15-agentic-rag/1-agenticrag.ipynb",
        {
            "stage": "LangGraph and agentic RAG",
            "title": "Agentic RAG: Retrieve, Grade, Rewrite Once",
            "difficulty": "Advanced",
            "key_idea": "Agentic RAG earns its loop only when a measurable retrieval failure triggers a bounded corrective action.",
            "summary": "This notebook builds an offline LangGraph that retrieves, grades lexical evidence, rewrites one vocabulary-mismatched query, and then answers with a citation. The retry budget is one; unsupported queries terminate by abstaining.",
            "why": "A fixed RAG chain cannot react when retrieval has no support. An unbounded agent can loop, spend, or hallucinate. Explicit grades, retry budgets, and terminal reasons make correction testable.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Retrieval, support grade, one rewrite, citation, abstention, trace | Hosted LLM, web search, arbitrary planning, hidden reasoning |",
            "mental_model": "```text\nquery -> retrieve -> supported? yes -> answer\n                    no  -> rewrite (max 1) -> retrieve -> answer/abstain\n```",
            "setup_code": r'''
from typing import TypedDict
import re
from langgraph.graph import END, START, StateGraph

DOCUMENTS = [
    {"id": "access", "text": "Role access requires manager approval."},
    {"id": "leave", "text": "Employees receive twenty days of paid leave each year."},
]

def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"a", "an", "do", "how", "is", "of", "the", "to"}

class RagState(TypedDict, total=False):
    question: str
    query: str
    document_id: str
    document_text: str
    score: int
    retries: int
    answer: str
    terminal_reason: str
''',
            "how_it_works": "Retrieval exposes a support score. The grader routes zero-score results to one deterministic rewrite; a second zero-score result abstains. The answer node can only use the selected document and includes its ID.",
            "baseline_text": "The raw query uses `holiday` while the source uses `paid leave`. A one-shot lexical retriever ties at zero and selects the access document by ID.",
            "baseline_code": r'''
question = "How much annual holiday do staff get?"

def retrieve_text(query: str) -> tuple[dict, int]:
    ranked = sorted(
        ((document, len(terms(query) & terms(document["text"]))) for document in DOCUMENTS),
        key=lambda item: (-item[1], item[0]["id"]),
    )
    return ranked[0]

baseline_document, baseline_score = retrieve_text(question)
baseline_document, baseline_score
''',
            "technique_text": "Each node returns a small state delta. The rewrite uses a governed alias, not an unconstrained answer. Routing depends on score and retry count, not prose reasoning.",
            "technique_code": r'''
def retrieve_node(state: RagState) -> RagState:
    document, score = retrieve_text(state.get("query", state["question"]))
    return {"document_id": document["id"], "document_text": document["text"], "score": score}

def route_after_retrieval(state: RagState) -> str:
    if state["score"] > 0: return "answer"
    if state.get("retries", 0) < 1: return "rewrite"
    return "abstain"

def rewrite_node(state: RagState) -> RagState:
    rewritten = state["question"].lower().replace("annual holiday", "paid leave").replace("staff", "employees")
    return {"query": rewritten, "retries": state.get("retries", 0) + 1}

def answer_node(state: RagState) -> RagState:
    return {"answer": f"{state['document_text']} [{state['document_id']}]", "terminal_reason": "supported"}

def abstain_node(state: RagState) -> RagState:
    return {"answer": "I do not know based on the available documents.", "terminal_reason": "unsupported"}
''',
            "experiment_text": "We compile the graph and compare the holiday question with an unsupported weather question. The trace must show at most one rewrite and a clear terminal reason.",
            "experiment_code": r'''
builder = StateGraph(RagState)
for name, node in (("retrieve", retrieve_node), ("rewrite", rewrite_node), ("answer", answer_node), ("abstain", abstain_node)):
    builder.add_node(name, node)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieval, {"answer": "answer", "rewrite": "rewrite", "abstain": "abstain"})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("answer", END)
builder.add_edge("abstain", END)
graph = builder.compile()

supported = graph.invoke({"question": question, "query": question, "retries": 0})
unsupported = graph.invoke({"question": "Will it rain tomorrow?", "query": "Will it rain tomorrow?", "retries": 0})
trace = list(graph.stream({"question": question, "query": question, "retries": 0}, stream_mode="updates"))
{"supported": supported, "unsupported": unsupported, "nodes": [next(iter(event)) for event in trace]}
''',
            "evaluation": "The holiday query follows `retrieve → rewrite → retrieve → answer`, retrieves `leave`, and cites it. The weather query uses its single rewrite budget and then abstains. Both terminate deterministically.",
            "checks_code": r'''
assert baseline_document["id"] == "access" and baseline_score == 0
assert supported["document_id"] == "leave" and supported["retries"] == 1 and "[leave]" in supported["answer"]
assert unsupported["terminal_reason"] == "unsupported" and unsupported["retries"] == 1
assert [next(iter(event)) for event in trace] == ["retrieve", "rewrite", "retrieve", "answer"]
print("Agentic RAG correction checks passed.")
''',
            "decision_guide": "| Failure | Response |\n|---|---|\n| Vocabulary mismatch | Bounded rewrite/expansion |\n| Corpus lacks answer | Abstain |\n| Known fixed sequence | Deterministic graph |\n| Several authorized strategies | Bounded router/agent with evaluation |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Infinite rewrite loop | No retry budget | Hard step/retry limit |\n| Wrong doc graded supported | Weak score/threshold | Labeled grader evaluation |\n| Unsupported answer | Answer ignores evidence | Evidence-only prompt/check |\n| Cost grows silently | Loop not traced | Per-node latency/token/tool budget |",
            "production_notes": "### Observability\nTrace query versions, ranked IDs/scores, grade, retry count, node sequence, citation, and terminal reason.\n\n### Safety and Guardrails\nRewrites inherit authorization scope and cannot introduce unrestricted sources.\n\n### Latency and Cost\nBudget the worst-case loop and keep a deterministic abstention path.",
            "practice": "Add a misspelling rewrite and prove the graph never executes more than one corrective pass.",
            "recall": "Toggle - Recall: When is a loop justified?\nWhen a measurable failure can trigger a bounded action that improves outcomes.\n\nToggle - Recall: What ends this graph?\nSupported evidence or an exhausted retry budget followed by abstention.",
            "sources": "- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic policy fixture",
            "confidence": "High for the bounded offline workflow",
            "next_review": "Evaluate learned graders and rewrite quality",
        },
    )


def build_agentic_router() -> None:
    build_standard_lesson(
        "Section-15-agentic-rag/1-agenticrag_1.ipynb",
        {
            "stage": "LangGraph and agentic RAG",
            "title": "Agentic RAG Routing: Retrieve, Calculate, or Abstain",
            "difficulty": "Intermediate",
            "key_idea": "Routing should choose the smallest authorized path that can answer the question, with an explicit abstention branch.",
            "summary": "This notebook implements a deterministic three-route controller: policy questions retrieve, arithmetic questions use a calculator, and unsupported weather questions abstain. A three-case routing table verifies path and answer behavior.",
            "why": "Sending every question through retrieval or a general agent adds irrelevant context, cost, and risk. Explicit route criteria make the system predictable and testable.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Route contract, retrieval/tool/abstain branches, routing accuracy | LLM router, ambiguous multi-intent queries, external tools |",
            "mental_model": "```text\nquestion -> classify route -> retrieve | calculate | abstain -> typed result\n```",
            "setup_code": r'''
import re

POLICY = "Employees receive twenty days of paid leave each year."

def route(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ("leave", "holiday", "policy")): return "retrieve"
    if re.fullmatch(r"\s*what is \d+ plus \d+\?\s*", lowered): return "calculate"
    return "abstain"
''',
            "how_it_works": "The router emits one of a closed set of labels. Each branch validates its own inputs and returns a common result shape containing route, answer, and optional citation.",
            "baseline_text": "A retrieve-everything baseline returns the leave policy even for arithmetic and weather questions, creating irrelevant context.",
            "baseline_code": r'''
def retrieve_everything(question: str) -> dict:
    return {"route": "retrieve", "answer": POLICY, "citation": "leave"}

[retrieve_everything(question) for question in ("How much leave?", "What is 5 plus 8?", "Will it rain?")]
''',
            "technique_text": "The branch implementation is intentionally small. In production the router can be learned, but the label set, branch contracts, fallback, and evaluation remain explicit.",
            "technique_code": r'''
def run(question: str) -> dict:
    selected = route(question)
    if selected == "retrieve":
        return {"route": selected, "answer": POLICY, "citation": "leave"}
    if selected == "calculate":
        left, right = map(int, re.findall(r"\d+", question))
        return {"route": selected, "answer": str(left + right), "citation": None}
    return {"route": selected, "answer": "I cannot answer with the available paths.", "citation": None}

run("How much annual holiday do employees get?")
''',
            "experiment_text": "We evaluate one labeled example per route. Route accuracy and output checks are separate so a correct route with a broken branch cannot pass.",
            "experiment_code": r'''
cases = [
    ("How much annual holiday do employees get?", "retrieve", "twenty"),
    ("What is 5 plus 8?", "calculate", "13"),
    ("Will it rain tomorrow?", "abstain", "cannot answer"),
]
results = [
    {"question": question, "expected_route": expected, "result": run(question), "answer_check": expected_text in run(question)["answer"].lower()}
    for question, expected, expected_text in cases
]
results
''',
            "evaluation": "All three examples select the expected route and pass branch-specific answer checks. This tiny table proves the controller contract, not general natural-language classification accuracy.",
            "checks_code": r'''
assert all(row["result"]["route"] == row["expected_route"] for row in results)
assert all(row["answer_check"] for row in results)
assert results[0]["result"]["citation"] == "leave"
assert results[1]["result"]["citation"] is None
print("Agentic routing checks passed.")
''',
            "decision_guide": "| Question | Route |\n|---|---|\n| Knowledge-base fact | Retrieve |\n| Deterministic arithmetic | Calculator/tool |\n| Unsupported/out of scope | Abstain |\n| Ambiguous multi-intent | Clarify or bounded plan |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Wrong branch | Router labels too vague | Labeled confusion matrix |\n| Correct branch, wrong output | Branch contract broken | Test branch separately |\n| Everything becomes agentic | No simple-route preference | Cost/risk-aware routing policy |\n| Unsupported route improvises | Missing fallback | Explicit abstention |",
            "production_notes": "### Observability\nLog route, confidence/rule, branch latency/cost, fallback, citation, and outcome label.\n\n### Safety and Guardrails\nAuthorize tools and sources after routing; classification is not permission.\n\n### Latency and Cost\nRoute simple deterministic work away from expensive model loops.",
            "practice": "Add a compound policy-plus-arithmetic question and decide whether to clarify or invoke a two-step plan.",
            "recall": "Toggle - Recall: What is the router's output?\nA closed, testable path label—not an answer.\n\nToggle - Recall: Why keep abstention?\nIt prevents unsupported questions from being forced through an irrelevant branch.",
            "sources": "- [LangGraph routing](https://docs.langchain.com/oss/python/langgraph/workflows-agents)\n- Repository-owned synthetic routing cases",
            "confidence": "High for the closed three-route controller",
            "next_review": "Add ambiguous and adversarial routing cases",
        },
    )


def react_spec(title: str) -> dict[str, str]:
    return {
        "stage": "LangGraph and agentic RAG",
        "title": title,
        "difficulty": "Advanced",
        "key_idea": "A retrieval-tool ReAct loop must validate search actions, preserve citations, and stop within a fixed budget.",
        "summary": "This notebook builds an offline LangGraph ReAct loop with one retrieval tool. The planner searches for paid-leave evidence, observes the cited result, and answers in one tool step; unsupported questions abstain without a call.",
        "why": "Turning retrieval into a tool gives an agent choice, but also introduces loop, argument, citation, and authorization failure modes that a fixed chain avoids.",
        "scope": "| Covers | Does not cover |\n|---|---|\n| Structured search action, tool validation, citation, abstention, step budget | Hosted planner, web search, multiple mutating tools, hidden reasoning |",
        "mental_model": "```text\nquestion -> plan search(query) -> retrieval observation -> plan final answer -> END\n```",
        "setup_code": r'''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

DOCUMENTS = {"leave": "Employees receive twenty days of paid leave each year."}
MAX_STEPS = 2

class State(TypedDict, total=False):
    question: str
    steps: int
    action: dict
    observation: dict
    answer: str
''',
        "how_it_works": "The planner proposes a typed `search` action only for an in-scope question. The action node validates the tool and query, returns content plus source ID, increments the budget, and loops once for answer construction.",
        "baseline_text": "A direct retrieval chain always searches, even for unsupported questions. It lacks a decision boundary and tool-step terminal reason.",
        "baseline_code": r'''
def always_search(question: str) -> dict:
    return {"content": DOCUMENTS["leave"], "source": "leave"}

always_search("Will it rain tomorrow?")
''',
        "technique_text": "The deterministic planner stands in for an LLM while preserving the operational contract. It emits actions, not free-form instructions, and never exposes hidden reasoning.",
        "technique_code": r'''
def plan(state: State) -> State:
    if state.get("observation"):
        item = state["observation"]
        return {"answer": f"Employees receive twenty days of paid leave [{item['source']}]."}
    if state.get("steps", 0) >= MAX_STEPS:
        return {"answer": "I stopped at the tool-step budget."}
    if "leave" in state["question"].lower() or "holiday" in state["question"].lower():
        return {"action": {"tool": "search", "query": "paid leave"}}
    return {"answer": "I cannot answer with the available retrieval tool."}

def act(state: State) -> State:
    action = state["action"]
    if action != {"tool": "search", "query": "paid leave"}:
        return {"answer": "Search action validation failed.", "action": {}}
    return {"observation": {"content": DOCUMENTS["leave"], "source": "leave"}, "steps": state.get("steps", 0) + 1, "action": {}}

def route(state: State) -> str:
    return "finish" if state.get("answer") else "act"
''',
        "experiment_text": "The answerable trace must be `plan → act → plan`, contain one search, and cite `leave`. The unsupported trace must terminate after the first plan.",
        "experiment_code": r'''
builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("act", act)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", route, {"finish": END, "act": "act"})
builder.add_edge("act", "plan")
agent = builder.compile()

answerable_trace = list(agent.stream({"question": "How much paid leave do employees get?", "steps": 0}, stream_mode="updates"))
answerable = agent.invoke({"question": "How much paid leave do employees get?", "steps": 0})
unsupported_trace = list(agent.stream({"question": "Will it rain tomorrow?", "steps": 0}, stream_mode="updates"))
{"answerable": answerable, "answerable_nodes": [next(iter(event)) for event in answerable_trace], "unsupported_nodes": [next(iter(event)) for event in unsupported_trace]}
''',
        "evaluation": "The leave question uses one validated search and answers with `[leave]`. The weather question terminates without retrieval. Both traces are bounded and inspectable.",
        "checks_code": r'''
assert answerable["steps"] == 1 and "[leave]" in answerable["answer"]
assert [next(iter(event)) for event in answerable_trace] == ["plan", "act", "plan"]
assert [next(iter(event)) for event in unsupported_trace] == ["plan"]
assert "cannot answer" in next(iter(unsupported_trace[0].values()))["answer"]
print("Retrieval-tool ReAct checks passed.")
''',
        "decision_guide": "| Need | Pattern |\n|---|---|\n| Retrieval always required | Fixed two-step RAG |\n| Retrieval optional among few tools | Bounded ReAct |\n| Known branching workflow | Explicit graph |\n| Unsupported question | Abstain |",
        "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Repeated searches | No observation/step rule | Budget and duplicate-action guard |\n| Citation lost | Tool returns text only | Typed result with source ID |\n| Prompt injects tool call | Free-form action | Schema, allowlist, authorization |\n| Weather retrieves leave policy | Missing abstention route | Scope classifier/fallback |",
        "production_notes": "### Observability\nTrace action schema, query, ranked IDs, source citation, step count, latency, and terminal reason.\n\n### Safety and Guardrails\nRetrieval filters and tenant authorization apply inside the tool, not only in the prompt.\n\n### Latency and Cost\nCap steps and duplicate calls; prefer fixed RAG when every valid question needs retrieval.",
        "practice": "Add a second document and require the agent to cite the exact retrieved source without exceeding two steps.",
        "recall": "Toggle - Recall: Why return source IDs from tools?\nThe final answer needs traceable evidence.\n\nToggle - Recall: When is ReAct unnecessary?\nWhen the retrieval-and-answer sequence is fixed for every request.",
        "sources": "- [ReAct paper](https://arxiv.org/abs/2210.03629)\n- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)",
        "confidence": "High for the offline retrieval-tool contract",
        "next_review": "Add multi-tool conflicts and typed retrieval errors",
    }


if __name__ == "__main__":
    build_agentic_workflow()
    build_agentic_router()
    build_standard_lesson("Section-15-agentic-rag/2-ReAct.ipynb", react_spec("ReAct RAG: A Bounded Retrieval Tool Loop"))
    copyfile(
        ROOT / "Section-15-agentic-rag/2-ReAct.ipynb",
        ROOT / "Section-15-agentic-rag/2-ReAct_1.ipynb",
    )
