"""Builds a ready-to-use SupportAgent (KB index + memory + calibrated config + graph)."""
import uuid

from agent.graph import Services, build_graph
from agent.guardrails import mask_pii
from agent.memory import JsonMemory
from config import MEMORY_FILE, load_rag_config
from knowledge_base.documents import KB_DOCUMENTS
from rag.store import VectorKB
from resilience.policies import invoke_with_global_timeout


class SupportAgent:
    def __init__(self, services: Services, **graph_kwargs):
        self.sv = services
        self.graph = build_graph(services, **graph_kwargs)

    def initial_state(self, session_id, query, trace_id=None):
        masked, found = mask_pii(query)  # mask at the edge: raw PII never enters state, checkpoints or logs
        return {"session_id": session_id, "trace_id": trace_id or uuid.uuid4().hex, "query": masked,
                "pii_found": found, "guardrails": [], "node_log": []}

    @staticmethod
    def config(thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def ask_full(self, session_id, query, trace_id=None, thread_id=None, global_timeout_s=None) -> dict:
        state = self.initial_state(session_id, query, trace_id)
        cfg = self.config(thread_id or state["trace_id"])
        if global_timeout_s:
            return invoke_with_global_timeout(self.graph, state, cfg, global_timeout_s)
        return self.graph.invoke(state, cfg)

    def ask(self, session_id, query, **kw) -> dict:
        return self.ask_full(session_id, query, **kw)["response"]


def build_agent(rebuild_index=True, memory_path=MEMORY_FILE, **graph_kwargs) -> SupportAgent:
    cfg = load_rag_config()
    kb = VectorKB(rebuild_docs=KB_DOCUMENTS if rebuild_index else None)
    if not rebuild_index:
        kb.docs = {d["doc_id"]: d for d in KB_DOCUMENTS}
    sv = Services(kb=kb, memory=JsonMemory(memory_path), strategy=cfg["strategy"],
                  threshold=cfg["thresholds"][cfg["strategy"]])
    return SupportAgent(sv, **graph_kwargs)
