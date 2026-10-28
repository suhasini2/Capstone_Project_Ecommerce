"""Tool 2: order lookup with a designed escalation score (Task 6).

escalation_score = 0.6 * delayed_shipment + 0.4 * min(days_since_created / 30, 1)
  - delay is the stronger signal (0.6); order age (older unresolved orders are more urgent) adds up to 0.4.
  - Recommend escalation when score > ESCALATION_THRESHOLD, the 80th percentile (nearest-rank) of the
    scores of *my generated dataset* -> roughly the top 20% of orders get escalated.
Known limitation: the score ignores status, so an old *Delivered* order can still score high.
"""
import math
from dataset import ORDERS, ORDERS_BY_ID

W_DELAY, W_AGE, MAX_DAYS = 0.6, 0.4, 30


def escalation_score(order: dict) -> float:
    age = min(order["days_since_created"] / MAX_DAYS, 1.0)
    return round(W_DELAY * float(order["delayed_shipment"]) + W_AGE * age, 4)


def percentile_nearest_rank(values: list, p: float) -> float:
    s = sorted(values)
    return s[max(math.ceil(p / 100 * len(s)) - 1, 0)]


ESCALATION_THRESHOLD = round(percentile_nearest_rank([escalation_score(o) for o in ORDERS], 80), 2)


def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order by ID and return its status, value and escalation score.

    Args:
        record_id: Order ID such as "NYK-1003".

    Returns:
        dict with record_id, status, order_value_inr, escalation_score (0-1),
        escalation_threshold and recommend_escalation; or {"error": ...} if the order does not exist.
    """
    rid = (record_id or "").strip().upper()
    order = ORDERS_BY_ID.get(rid)
    if order is None:
        return {"record_id": rid, "error": f"No order found with ID {rid!r}."}
    score = escalation_score(order)
    return {"record_id": rid, "status": order["status"], "order_value_inr": order["order_value_inr"],
            "escalation_score": score, "escalation_threshold": ESCALATION_THRESHOLD,
            "recommend_escalation": score > ESCALATION_THRESHOLD}


if __name__ == "__main__":
    scores = sorted(escalation_score(o) for o in ORDERS)
    print("formula: escalation_score = 0.6*delayed_shipment + 0.4*min(days_since_created/30, 1)")
    print(f"score range in dataset: {scores[0]} .. {scores[-1]}")
    print(f"threshold (80th percentile of dataset scores) = {ESCALATION_THRESHOLD}")
    print(f"orders recommended for escalation: {sum(s > ESCALATION_THRESHOLD for s in scores)} of {len(scores)}")
    for rid in ("NYK-1003", ORDERS[-1]["record_id"], "NYK-9999"):
        print(rid, "->", check_order_status(rid))
