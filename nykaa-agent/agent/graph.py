"""LangGraph orchestration (Tasks 7-10, 15, 16).

START -> memory_load -> input_guard -> router --(conditional edge)--> rag_node -> output_guard -> finalize
                                               |-> order_node ----------------------------------> finalize
                                               '-> finalize   (blocked: prompt injection)
"""
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.guardrails import detect_injection, groundedness_check, mask_pii
from agent.memory import JsonMemory
from agent.router import classify_intent
from agent.schema import validate_response
from agent.tools import check_order_status
from rag.generate import answer_query
from resilience.policies import GlobalTimeoutError, log_execution, make_retry_policy, run_with_timeout

REFUSAL = "I'm sorry, I can't answer that reliably from our policy documents. I can connect you with a human agent."
BLOCKED = "I can't help with that request. Please ask a question about Nykaa orders or policies."
ASK_ID = "Which order do you mean? Please share your order ID (for example NYK-1003)."


class AgentState(TypedDict, total=False):
    session_id: str
    trace_id: str
    query: str                       # always the MASKED query
    pii_found: list
    last_record_id: Optional[str]
    injection: Optional[str]
    intent: str
    record_id: Optional[str]
    contexts: list
    top_score: Optional[float]
    draft: str
    answer: str
    sources: list
    grounded: bool
    refused: bool
    escalation: Optional[dict]
    guardrails: Annotated[list, operator.add]
    node_log: Annotated[list, operator.add]
    deadline: Optional[float]
    response: dict


@dataclass
class Services:
    kb: object
    memory: JsonMemory
    strategy: str
    threshold: float


def build_graph(sv: Services, checkpointer=None, interrupt_before=None, faults=None,
                node_timeouts=None, exec_log_path=None):
    node_timeouts = node_timeouts or {}

    def guarded(name, fn):
        """Adds: global-deadline check, execution log, fault injection, optional per-node timeout."""
        def node(state):
            dl = state.get("deadline")
            if dl is not None and time.time() > dl:
                raise GlobalTimeoutError(f"global deadline passed before node '{name}'")
            log_execution(exec_log_path, name)

            def call():
                if faults:
                    faults.before(name)
                return fn(state)
            return run_with_timeout(call, node_timeouts[name], name) if name in node_timeouts else call()
        return node

    def memory_load(state):
        s = sv.memory.load(state["session_id"])
        return {"last_record_id": s.get("last_record_id"), "node_log": ["memory_load"]}

    def input_guard(state):
        text, found = mask_pii(state["query"])
        inj = detect_injection(text)
        tags = [f"pii_masked:{t}" for t in sorted(set(found) | set(state.get("pii_found", [])))]
        if inj:
            tags.append("prompt_injection_blocked")
        return {"query": text, "injection": inj, "guardrails": tags, "node_log": ["input_guard"]}

    def router(state):
        if state.get("injection"):
            return {"intent": "blocked", "node_log": ["router"]}
        intent, rid = classify_intent(state["query"], state.get("last_record_id"))
        return {"intent": intent, "record_id": rid, "node_log": ["router"]}

    def route_after_router(state):  # the conditional edge
        return {"policy": "rag_node", "order_status": "order_node"}.get(state["intent"], "finalize")

    def rag_node(state):
        res = answer_query(sv.kb, sv.strategy, state["query"], sv.threshold)
        ctx = [{"doc_id": h["doc_id"], "text": h["text"], "score": round(float(h["score"]), 4)} for h in res["hits"]]
        return {"draft": res["answer"], "contexts": ctx, "top_score": round(float(res["top_score"]), 4),
                "node_log": ["rag_node"]}

    def order_node(state):
        rid = state.get("record_id")
        if not rid:
            return {"answer": ASK_ID, "sources": [], "grounded": True, "refused": False,
                    "escalation": None, "node_log": ["order_node"]}
        info = check_order_status(rid)
        if "error" in info:
            return {"answer": info["error"], "sources": ["orders_db"], "grounded": True, "refused": False,
                    "escalation": None, "node_log": ["order_node"]}
        verdict = "recommend escalating to a human agent" if info["recommend_escalation"] else "no escalation needed"
        answer = (f"Order {rid} is {info['status']} (value INR {info['order_value_inr']}). Escalation score "
                  f"{info['escalation_score']:.2f} vs threshold {info['escalation_threshold']:.2f}: {verdict}.")
        return {"answer": answer, "sources": [f"orders_db:{rid}"], "grounded": True, "refused": False,
                "escalation": {"score": info["escalation_score"], "threshold": info["escalation_threshold"],
                               "recommended": info["recommend_escalation"]},
                "node_log": ["order_node"]}

    def output_guard(state):
        texts = [c["text"] for c in state.get("contexts", [])]
        ok, reason = groundedness_check(state.get("draft", ""), texts, state.get("top_score"), sv.threshold)
        if ok:
            return {"answer": state["draft"], "sources": sorted({c["doc_id"] for c in state["contexts"]}),
                    "grounded": True, "refused": False, "node_log": ["output_guard"]}
        return {"answer": REFUSAL, "sources": [], "grounded": False, "refused": True,
                "guardrails": ["groundedness_refusal"], "node_log": ["output_guard"]}

    def finalize(state):
        intent = state["intent"]
        if intent == "blocked":
            state = {**state, "answer": BLOCKED, "sources": [], "grounded": False, "refused": True, "escalation": None}
        resp = {"trace_id": state["trace_id"], "session_id": state["session_id"], "query": state["query"],
                "intent": intent, "answer": state["answer"], "sources": state.get("sources", []),
                "grounded": state.get("grounded", False), "refused": state.get("refused", False),
                "guardrails": state.get("guardrails", []), "escalation": state.get("escalation"),
                "top_score": state.get("top_score")}
        validate_response(resp)  # raises if the contract is broken
        found_rid = state.get("record_id") if (intent == "order_status" and state.get("escalation")) else None
        sv.memory.append_turn(state["session_id"], state["query"], state["answer"], intent, found_rid)
        return {"response": resp, "node_log": ["finalize"]}

    b = StateGraph(AgentState)
    retry = make_retry_policy()
    for name, fn, rp in [("memory_load", memory_load, None), ("input_guard", input_guard, None),
                         ("router", router, None), ("rag_node", rag_node, retry), ("order_node", order_node, retry),
                         ("output_guard", output_guard, None), ("finalize", finalize, None)]:
        b.add_node(name, guarded(name, fn), retry_policy=rp) if rp else b.add_node(name, guarded(name, fn))
    b.add_edge(START, "memory_load")
    b.add_edge("memory_load", "input_guard")
    b.add_edge("input_guard", "router")
    b.add_conditional_edges("router", route_after_router,
                            {"rag_node": "rag_node", "order_node": "order_node", "finalize": "finalize"})
    b.add_edge("rag_node", "output_guard")
    b.add_edge("output_guard", "finalize")
    b.add_edge("order_node", "finalize")
    b.add_edge("finalize", END)
    return b.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
