"""Task 15: SQLite checkpointing - stop mid-run, resume the SAME thread, prove nothing re-ran.
Run: python -m resilience.checkpoint_demo"""
import json
import sqlite3
from collections import Counter

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.factory import build_agent
from config import ARTIFACTS, CHECKPOINT_DB

EXEC_LOG = ARTIFACTS / "exec_log.jsonl"
THREAD = "demo-thread-1"


def counts():
    return dict(Counter(json.loads(l)["node"] for l in EXEC_LOG.read_text().splitlines()))


def main():
    EXEC_LOG.unlink(missing_ok=True)
    CHECKPOINT_DB.unlink(missing_ok=True)
    query = "How long does a COD refund take?"

    # ---- (a)+(b): run, deliberately stopped before rag_node
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    agent = build_agent(checkpointer=SqliteSaver(conn), interrupt_before=["rag_node"], exec_log_path=EXEC_LOG)
    agent.graph.invoke(agent.initial_state("ckpt-session", query), agent.config(THREAD))
    snap = agent.graph.get_state(agent.config(THREAD))
    print(f"[phase 1] thread={THREAD}  query={query!r}")
    print(f"[phase 1] nodes completed so far : {snap.values['node_log']}")
    print(f"[phase 1] next node(s) pending   : {snap.next}   <- run stopped BEFORE rag_node")
    print(f"[phase 1] node executions        : {counts()}")
    conn.close()
    del agent
    print("\n--- simulated crash: connection closed, graph + agent objects destroyed ---\n")

    # ---- (c): brand-new connection/graph, resume the same thread
    conn2 = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    agent2 = build_agent(rebuild_index=False, checkpointer=SqliteSaver(conn2), exec_log_path=EXEC_LOG)
    snap2 = agent2.graph.get_state(agent2.config(THREAD))
    print(f"[phase 2] loaded from checkpoint : node_log={snap2.values['node_log']} next={snap2.next}")
    out = agent2.graph.invoke(None, agent2.config(THREAD))
    print(f"[phase 2] run completed. full node_log: {out['node_log']}")
    print(f"[phase 2] answer: {out['response']['answer']}")
    c = counts()
    print(f"[phase 2] node executions        : {c}")
    for n in ("memory_load", "input_guard", "router"):
        assert c[n] == 1, f"{n} was re-executed!"
    print("[proof] memory_load, input_guard and router each executed exactly once -> restored from checkpoint, NOT re-run.")


if __name__ == "__main__":
    main()
