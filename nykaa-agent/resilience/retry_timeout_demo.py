"""Task 16: retry with backoff, per-node timeout, global timeout.  Run: python -m resilience.retry_timeout_demo"""
import time

from agent.factory import build_agent
from resilience import policies as P

Q = "How long does a COD refund take?"


def main():
    print(f"retry policy: max_attempts={P.MAX_ATTEMPTS}, initial_interval={P.INITIAL_INTERVAL}s, "
          f"backoff_factor={P.BACKOFF_FACTOR}, max_interval={P.MAX_INTERVAL}s, jitter={P.JITTER}")

    # (a) transient failure recovered by retry
    f = P.FaultInjector(); f.fail_first("rag_node", 2)
    agent = build_agent(faults=f)
    t0 = time.time()
    r = agent.ask("retry-demo", Q)
    print(f"\n(a) rag_node failed twice (simulated), succeeded on attempt #{f.calls['rag_node']} "
          f"after {time.time()-t0:.2f}s -> answer: {r['answer'][:70]}...")
    assert f.calls["rag_node"] == 3

    # (b) per-node timeout fires a clean error
    f = P.FaultInjector(); f.slow("rag_node", 3.0)
    agent = build_agent(rebuild_index=False, faults=f, node_timeouts={"rag_node": 0.5})
    t0 = time.time()
    try:
        agent.ask("timeout-demo", Q)
        raise SystemExit("expected a timeout!")
    except P.NodeTimeoutError as e:
        print(f"\n(b) per-node timeout fired cleanly after {time.time()-t0:.2f}s (not a hang): {type(e).__name__}: {e}")

    # (c) global timeout cancels the whole run
    f = P.FaultInjector()
    for n in ("memory_load", "input_guard", "router", "rag_node"):
        f.slow(n, 0.4)
    agent = build_agent(rebuild_index=False, faults=f)
    t0 = time.time()
    try:
        agent.ask("global-demo", Q, global_timeout_s=0.7)
        raise SystemExit("expected a global timeout!")
    except P.GlobalTimeoutError as e:
        print(f"\n(c) global timeout fired after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
        time.sleep(1.2)  # let the abandoned run hit its next node boundary
        print(f"    abandoned run stopped at a node boundary; rag_node calls = {f.calls['rag_node']} (never reached)")


if __name__ == "__main__":
    main()
