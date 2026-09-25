#!/usr/bin/env python3
"""Rebuild multimodal and LangGraph foundation lessons."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rebuild_ingestion_lessons import build_standard_lesson  # noqa: E402


def build_multimodal() -> None:
    build_standard_lesson(
        "11-multiModal-multi-modal-rag/multimodalopenai.ipynb",
        {
            "stage": "Multimodal RAG",
            "title": "Multimodal PDF RAG: Text and Image Evidence",
            "difficulty": "Intermediate",
            "key_idea": "A multimodal pipeline keeps text and image evidence distinct, then fuses conclusions with page-level provenance instead of pretending every fact came from OCR text.",
            "summary": "This notebook parses a one-page revenue PDF offline. Text extraction states that Q3 grew most; image analysis detects three colored bars with increasing heights. The modalities independently support the trend but neither provides exact revenue values.",
            "why": "Text-only parsing misses chart geometry, while image-only interpretation can miss captions and qualifications. Reliable answers need modality-aware records, provenance, and explicit limits.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| PDF text, embedded-image extraction, simple bar geometry, modality fusion, page citations | General vision model, OCR, arbitrary chart understanding, exact unlabeled values |",
            "mental_model": "```text\nPDF page -> text blocks --------\\\n           image assets -> vision/geometry -> fused evidence -> answer + page citation\n```",
            "setup_code": r'''
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import fitz
from PIL import Image
from rag_101 import find_repo_root

REPO_ROOT = find_repo_root()
PDF_PATH = REPO_ROOT / "11-multiModal-multi-modal-rag/multimodal_sample.pdf"
pdf = fitz.open(PDF_PATH)
page = pdf[0]
len(pdf), len(page.get_text()), len(page.get_images(full=True))
''',
            "how_it_works": "We emit one text record and one image record from the same page. The image record carries dimensions, checksum, and parent page. A deliberately narrow geometry function finds the three saturated colored bars and compares heights; it is not a universal chart parser.",
            "baseline_text": "The text-only baseline can answer which quarter grew most because the prose says Q3 had the highest growth, but it cannot independently verify the visual trend.",
            "baseline_code": r'''
page_text = page.get_text().strip()
text_record = {
    "id": "revenue:p1:text", "source": PDF_PATH.relative_to(REPO_ROOT).as_posix(),
    "page": 1, "modality": "text", "content": page_text,
}
{"characters": len(page_text), "mentions_q3_highest": "highest growth recorded in Q3" in page_text}
''',
            "technique_text": "The image is extracted without saving a machine-specific path. Pixels are classified by dominant red, green, or blue channel; each color's vertical extent gives a bar height. The method is valid only for this fixture's simple unlabeled chart.",
            "technique_code": r'''
xref = page.get_images(full=True)[0][0]
image_payload = pdf.extract_image(xref)
image_bytes = image_payload["image"]
image = Image.open(BytesIO(image_bytes)).convert("RGB")
pixels = image.load()

def bar_height(channel: int) -> int:
    points = [
        (x, y) for y in range(image.height) for x in range(image.width)
        if pixels[x, y][channel] > 100 and all(pixels[x, y][other] < 80 for other in range(3) if other != channel)
    ]
    return max(y for _, y in points) - min(y for _, y in points) + 1

bar_heights = {"Q1": bar_height(2), "Q2": bar_height(1), "Q3": bar_height(0)}
image_record = {
    "id": "revenue:p1:image1", "source": text_record["source"], "page": 1,
    "modality": "image", "sha256": sha256(image_bytes).hexdigest(),
    "width": image.width, "height": image.height, "observations": bar_heights,
}
image_record
''',
            "experiment_text": "We compare the prose claim with the image-derived ordering. Agreement supports the qualitative trend; absence of labels prevents a defensible numeric answer.",
            "experiment_code": r'''
image_order = sorted(bar_heights, key=bar_heights.get)
text_claim = "Q3"
image_claim = image_order[-1]
results = {
    "text_highest_quarter": text_claim,
    "image_highest_quarter": image_claim,
    "modalities_agree": text_claim == image_claim,
    "bar_heights_pixels": bar_heights,
    "exact_revenue_available": False,
}
results
''',
            "evaluation": "Text and image agree that **Q3 is highest**, and the detected bar heights rise from Q1 to Q3. Because the chart has no labels or axis values, the system must not invent exact revenue numbers. The page citation remains `multimodal_sample.pdf`, page 1.",
            "checks_code": r'''
assert results["modalities_agree"]
assert image_order == ["Q1", "Q2", "Q3"]
assert all(value > 0 for value in bar_heights.values())
assert not results["exact_revenue_available"]
assert text_record["page"] == image_record["page"] == 1
pdf.close()
print("Multimodal extraction and fusion checks passed.")
''',
            "decision_guide": "| Evidence | Path |\n|---|---|\n| Narrative prose | Text parser/retriever |\n| Labeled chart/table | Layout or chart extraction |\n| Unlabeled visual trend | Vision/geometry with qualitative caveat |\n| Cross-modal claim | Retrieve both and reconcile provenance |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Chart missing | Text-only ingestion | Extract/page-link images |\n| Exact values invented | Unlabeled pixels treated as data | Abstain on numbers |\n| Text and image disagree | Version/layout/extraction issue | Surface conflict and inspect source |\n| Citation points to wrong page | Parent link dropped | Inherit source/page on every asset |",
            "production_notes": "### Observability\nTrack pages, text blocks, images, extraction method, modality agreement, model/version, and unresolved conflicts.\n\n### Safety and Guardrails\nImages can contain sensitive data and prompt-like text; apply the same authorization and redaction policy as prose.\n\n### Latency and Cost\nRoute text-only questions away from vision; cache image features by content hash.",
            "practice": "Add axis labels to the fixture and specify what extra evidence is required before reporting numeric values.",
            "recall": "Toggle - Recall: Why keep modalities separate?\nTheir extraction methods, confidence, and failure modes differ.\n\nToggle - Recall: What can this image prove?\nOnly the relative bar ordering for this simple fixture, not exact revenue.",
            "sources": "- [PyMuPDF image extraction](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)\n- Repository fixture: `multimodal_sample.pdf`",
            "confidence": "High for the fixture-specific qualitative check",
            "next_review": "Add labels, OCR, and cross-modal conflict fixtures",
        },
    )


def build_streaming() -> None:
    build_standard_lesson(
        "section-14-agents-architecture/streaming.ipynb",
        {
            "stage": "LangGraph and agentic foundations",
            "title": "LangGraph Streaming: State Snapshots and Node Updates",
            "difficulty": "Intermediate",
            "key_idea": "Streaming is an event contract. Consumers must know whether each event is a full state snapshot, a node update, or a model token.",
            "summary": "This notebook builds a two-node offline LangGraph and compares `stream_mode='values'` with `stream_mode='updates'`. It reconstructs final state from updates and verifies parity with normal invocation.",
            "why": "Streaming improves perceived latency and observability, but confusing deltas with full state causes missing fields, duplicate rendering, and fragile clients.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Typed state, deterministic nodes, values/updates modes, reconstruction, cancellation notes | Hosted token stream, UI framework, distributed backpressure |",
            "mental_model": "```text\ninput state -> prepare node -> answer node -> final state\n                | updates       | updates\n                `------ values snapshots ------`\n```",
            "setup_code": r'''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict, total=False):
    text: str
    normalized: str
    answer: str

def prepare(state: State) -> State:
    return {"normalized": state["text"].strip().lower()}

def answer(state: State) -> State:
    return {"answer": f"Echo: {state['normalized']}"}
''',
            "how_it_works": "`values` emits the accumulated state after each transition, including the initial input. `updates` emits only each node's returned delta, keyed by node name. Token streaming is a separate provider/model capability and should not be conflated with state streaming.",
            "baseline_text": "Normal invocation returns only final state. It is the correctness reference but exposes no intermediate progress.",
            "baseline_code": r'''
builder = StateGraph(State)
builder.add_node("prepare", prepare)
builder.add_node("answer", answer)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "answer")
builder.add_edge("answer", END)
graph = builder.compile()

input_state = {"text": "  Hello Streaming  "}
final_state = graph.invoke(input_state)
final_state
''',
            "technique_text": "We collect both stream modes from the same input. The notebook keeps event lists short and inspects their shape rather than printing long model traces.",
            "technique_code": r'''
value_events = list(graph.stream(input_state, stream_mode="values"))
update_events = list(graph.stream(input_state, stream_mode="updates"))
{"values": value_events, "updates": update_events}
''',
            "experiment_text": "A client reconstructs state by starting with the input and merging each node update. The reconstructed state must equal both the final `values` event and ordinary invocation.",
            "experiment_code": r'''
reconstructed = dict(input_state)
for event in update_events:
    for node_update in event.values():
        reconstructed.update(node_update)

results = {
    "value_event_count": len(value_events),
    "update_event_count": len(update_events),
    "final_value": value_events[-1],
    "reconstructed": reconstructed,
    "invoke_result": final_state,
}
results
''',
            "evaluation": "The graph emits **3 value snapshots** (input plus two nodes) and **2 update events**. Merging updates reconstructs exactly the same final state as `invoke`. This proves state-event semantics for the local graph, not network token behavior.",
            "checks_code": r'''
assert results["value_event_count"] == 3
assert results["update_event_count"] == 2
assert results["reconstructed"] == results["final_value"] == results["invoke_result"]
assert [next(iter(event)) for event in update_events] == ["prepare", "answer"]
print("LangGraph streaming checks passed.")
''',
            "decision_guide": "| Consumer need | Mode |\n|---|---|\n| Render current complete workflow state | `values` |\n| Observe node-by-node deltas | `updates` |\n| Show model text as generated | Token/message streaming |\n| Durable audit | Persist selected structured events |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Fields disappear | Update treated as full state | Merge by documented reducer |\n| UI duplicates content | Snapshots appended as deltas | Replace vs append by event type |\n| Work continues after disconnect | Cancellation not propagated | Cooperative cancellation/timeouts |\n| Producer overwhelms client | No backpressure/bounds | Buffer limits and coalescing |",
            "production_notes": "### Observability\nInclude run/thread ID, node, sequence, event type, timestamps, and terminal status.\n\n### Safety and Guardrails\nDo not stream hidden prompts, secrets, or unrestricted tool payloads.\n\n### Latency and Cost\nStreaming changes delivery, not model compute cost; measure time-to-first-event and total duration separately.",
            "practice": "Add a third node that records string length, then update the expected event counts and reconstruction check.",
            "recall": "Toggle - Recall: What does `updates` emit?\nEach node's state delta keyed by node name.\n\nToggle - Recall: Does streaming reduce total compute?\nNot necessarily; it mainly improves progressive delivery and observability.",
            "sources": "- [LangGraph streaming documentation](https://docs.langchain.com/oss/python/langgraph/streaming)\n- [LangGraph graph API](https://reference.langchain.com/python/langgraph/graphs/)",
            "confidence": "High for the installed LangGraph state modes",
            "next_review": "Add token streaming, cancellation, and backpressure fixtures",
        },
    )


def build_react_foundation() -> None:
    build_standard_lesson(
        "section-14-agents-architecture/ReActAgents.ipynb",
        {
            "stage": "LangGraph and agentic foundations",
            "title": "ReAct Agents: A Bounded Tool Loop",
            "difficulty": "Intermediate",
            "key_idea": "A ReAct agent is a state machine around tool calls. Tool contracts, routing, step budgets, and terminal states matter more than hidden reasoning text.",
            "summary": "This notebook builds an offline LangGraph tool loop. A deterministic planner routes an addition question to a calculator, records the observation, and finishes in one tool step; unsupported questions abstain without calling a tool.",
            "why": "Agent demos often hide termination and error behavior inside an LLM. Making plan, action, observation, and finish states explicit makes loops testable and prevents unbounded tool use.",
            "scope": "| Covers | Does not cover |\n|---|---|\n| Typed state, tool contract, routing, observation, abstention, step budget | Hosted LLM planner, web tools, hidden chain-of-thought, long-term memory |",
            "mental_model": "```text\nquestion -> plan -> tool action -> observation -> plan -> answer/abstain\n                    ^ bounded by step budget -----------|\n```",
            "setup_code": r'''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class AgentState(TypedDict, total=False):
    question: str
    steps: int
    action: str
    observation: str
    answer: str

MAX_TOOL_STEPS = 2

def calculator_add(left: str, right: str) -> str:
    return str(int(left) + int(right))
''',
            "how_it_works": "The planner emits a structured action or terminal answer. The action node validates the tool name/arguments, records an observation, increments the budget, and returns to planning. No hidden rationale is required for control flow.",
            "baseline_text": "A direct hard-coded answer would be fast but cannot expose which tool ran, enforce a common budget, or generalize to several controlled actions.",
            "baseline_code": r'''
def direct_answer(question: str) -> str:
    return "13" if question == "What is 5 plus 8?" else "I do not know."

direct_answer("What is 5 plus 8?"), direct_answer("What is today's weather?")
''',
            "technique_text": "The planner is deterministic so the lesson runs offline. Replacing it with an LLM changes how actions are proposed, not the validation, execution, budget, or terminal-state responsibilities.",
            "technique_code": r'''
def plan(state: AgentState) -> AgentState:
    if state.get("observation"):
        return {"answer": f"The result is {state['observation']}."}
    if state.get("steps", 0) >= MAX_TOOL_STEPS:
        return {"answer": "I stopped after reaching the tool-step budget."}
    if state["question"].strip().lower() == "what is 5 plus 8?":
        return {"action": "calculator:add:5:8"}
    return {"answer": "I cannot answer with the available tools."}

def act(state: AgentState) -> AgentState:
    tool, operation, left, right = state["action"].split(":")
    if (tool, operation) != ("calculator", "add"):
        return {"answer": "Tool validation failed.", "action": ""}
    return {
        "observation": calculator_add(left, right),
        "steps": state.get("steps", 0) + 1,
        "action": "",
    }

def route(state: AgentState) -> str:
    return "finish" if state.get("answer") else "act"
''',
            "experiment_text": "We compile the loop, run an answerable and unsupported question, and inspect update events. Success requires one validated tool step for arithmetic and zero tool steps for abstention.",
            "experiment_code": r'''
builder = StateGraph(AgentState)
builder.add_node("plan", plan)
builder.add_node("act", act)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", route, {"finish": END, "act": "act"})
builder.add_edge("act", "plan")
agent = builder.compile()

answerable = agent.invoke({"question": "What is 5 plus 8?", "steps": 0})
unsupported = agent.invoke({"question": "What is today's weather?", "steps": 0})
trace = list(agent.stream({"question": "What is 5 plus 8?", "steps": 0}, stream_mode="updates"))
{"answerable": answerable, "unsupported": unsupported, "trace": trace}
''',
            "evaluation": "The arithmetic run follows `plan → act → plan`, calls one tool, observes `13`, and terminates. The weather question abstains in the first planning node with zero tool calls. The trace exposes control state without storing private chain-of-thought.",
            "checks_code": r'''
assert answerable["answer"] == "The result is 13." and answerable["steps"] == 1
assert unsupported["answer"] == "I cannot answer with the available tools." and unsupported["steps"] == 0
assert [next(iter(event)) for event in trace] == ["plan", "act", "plan"]
assert trace[1]["act"]["observation"] == "13"
print("Bounded ReAct checks passed.")
''',
            "decision_guide": "| Need | Architecture |\n|---|---|\n| Fixed known sequence | Deterministic chain/graph |\n| One bounded tool choice | Single ReAct loop |\n| Complex dependencies | Explicit workflow planner |\n| No authorized supporting tool | Abstain, do not improvise |",
            "failure_modes": "| Symptom | Cause | Fix |\n|---|---|---|\n| Infinite loop | No terminal condition/budget | Step/time/tool budgets |\n| Wrong tool arguments | Free-form action | Structured schema and validation |\n| Tool error becomes answer | Observation unchecked | Typed error branch/retry policy |\n| Conversation leaks across users | Memory scope wrong | Explicit thread/tenant boundary |",
            "production_notes": "### Observability\nTrace node, action schema, safe arguments, tool result status, step count, latency, terminal reason, and model/version.\n\n### Safety and Guardrails\nAllowlist tools, validate arguments, authorize every call, and require confirmation for consequential actions.\n\n### Latency and Cost\nEach loop step may add model and tool calls; enforce budgets and prefer deterministic workflows when the route is known.",
            "practice": "Add a divide tool with zero-division handling and prove that the loop terminates with a typed error observation.",
            "recall": "Toggle - Recall: What makes the loop safe to operate?\nStructured actions, validation, authorization, budgets, and terminal states.\n\nToggle - Recall: Why avoid storing hidden reasoning?\nControl flow can be observed through actions and state without collecting sensitive internal rationale.",
            "sources": "- [ReAct paper](https://arxiv.org/abs/2210.03629)\n- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)",
            "confidence": "High for the deterministic bounded loop",
            "next_review": "Add typed tool errors, retries, and checkpoint boundaries",
        },
    )


if __name__ == "__main__":
    build_multimodal()
    build_streaming()
    build_react_foundation()
