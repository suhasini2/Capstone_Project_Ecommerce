"""Retry / timeout building blocks (Task 16)."""
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Exponential-backoff retry policy (stated values, used by LangGraph's RetryPolicy):
MAX_ATTEMPTS = 4          # 1 try + 3 retries
INITIAL_INTERVAL = 0.2    # seconds before the first retry
BACKOFF_FACTOR = 2.0      # 0.2 -> 0.4 -> 0.8 ...
MAX_INTERVAL = 1.0        # cap on any single wait
JITTER = True             # randomises each wait so retries don't synchronise


class TransientError(Exception):
    """A failure worth retrying (network blip, upstream 503...)."""


class NodeTimeoutError(Exception):
    """A single node exceeded its time budget."""


class GlobalTimeoutError(Exception):
    """The whole graph run exceeded its total time budget."""


def make_retry_policy():
    from langgraph.types import RetryPolicy
    return RetryPolicy(max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_INTERVAL, backoff_factor=BACKOFF_FACTOR,
                       max_interval=MAX_INTERVAL, jitter=JITTER, retry_on=TransientError)


def run_with_timeout(fn, seconds: float, name: str):
    """Per-node timeout: raises NodeTimeoutError instead of hanging.
    (A Python thread cannot be force-killed; the abandoned call finishes in the background.)"""
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except FuturesTimeout:
        raise NodeTimeoutError(f"node '{name}' exceeded its {seconds}s timeout") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def invoke_with_global_timeout(graph, state_in: dict, config: dict, seconds: float):
    """Global timeout: the caller gets GlobalTimeoutError at the deadline, and every node checks the
    shared `deadline` in state before starting, so the abandoned run stops at the next node boundary."""
    state_in = {**state_in, "deadline": time.time() + seconds}
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(graph.invoke, state_in, config)
    try:
        return fut.result(timeout=seconds)
    except FuturesTimeout:
        raise GlobalTimeoutError(f"graph run exceeded the {seconds}s global timeout") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class FaultInjector:
    """Simulated failures for demos: slow(node, s) sleeps; fail_first(node, n) raises TransientError n times."""
    def __init__(self):
        self.calls, self._slow, self._fail = defaultdict(int), {}, {}

    def slow(self, node, seconds): self._slow[node] = seconds
    def fail_first(self, node, n): self._fail[node] = n

    def before(self, node):
        self.calls[node] += 1
        if node in self._slow:
            time.sleep(self._slow[node])
        if self.calls[node] <= self._fail.get(node, 0):
            raise TransientError(f"simulated transient failure in '{node}' (call #{self.calls[node]})")


def log_execution(path, node: str) -> None:
    """Append one line per node *start*; lets a demo prove which nodes really executed (survives restarts)."""
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"node": node, "ts": round(time.time(), 3)}) + "\n")
