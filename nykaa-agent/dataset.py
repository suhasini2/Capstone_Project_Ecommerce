"""Seeded, deterministic order dataset (Task 1).

Design choices (also stated in README.md so a grader can reproduce the data):
  * SEED = 42, NUM_ORDERS = 48
  * category weights   : Apparel .22, Electronics .18, Home .15, Footwear .20, Beauty .25
  * status weights     : Placed .15, Shipped .25, Delivered .40, Returned .10, Refunded .10
  * delay probability  : 0.20 per order
  * order value (INR)  : per-category ranges below. Reasoning: a Nykaa basket ranges from a
    Rs 199 lip product to a Rs 40,000 electronics item, so each category gets its own
    realistic band instead of one flat range.
Records are never hand-edited: if validation fails, change SEED / weights and regenerate.
"""
import random
from collections import Counter

SEED = 42
NUM_ORDERS = 48
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
CATEGORY_WEIGHTS = [0.22, 0.18, 0.15, 0.20, 0.25]
STATUS_WEIGHTS = [0.15, 0.25, 0.40, 0.10, 0.10]
PRICE_RANGE_INR = {
    "Apparel": (499, 5999),
    "Electronics": (999, 39999),
    "Home": (399, 12999),
    "Footwear": (699, 7999),
    "Beauty": (199, 3999),
}
DELAY_PROBABILITY = 0.20


def generate_orders(seed: int = SEED, n: int = NUM_ORDERS) -> list:
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        lo, hi = PRICE_RANGE_INR[category]
        orders.append({
            "record_id": f"NYK-{1001 + i}",
            "category": category,
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0],
            "order_value_inr": rng.randint(lo, hi),
            "days_since_created": rng.randint(0, 30),
            "delayed_shipment": rng.random() < DELAY_PROBABILITY,
        })
    return orders


def validate_orders(orders: list) -> dict:
    """Assert every structural threshold from the brief; return the report."""
    cats = Counter(o["category"] for o in orders)
    stats = Counter(o["status"] for o in orders)
    delayed_pct = 100.0 * sum(o["delayed_shipment"] for o in orders) / len(orders)
    assert len(orders) >= 40, "need >= 40 records"
    assert all(cats[c] >= 3 for c in CATEGORIES), f"each category needs >= 3: {dict(cats)}"
    assert all(stats[s] >= 1 for s in STATUSES), f"each status needs >= 1: {dict(stats)}"
    assert 10.0 <= delayed_pct <= 30.0, f"delayed % {delayed_pct:.1f} outside 10-30"
    assert all(0 <= o["days_since_created"] <= 30 for o in orders)
    return {"total": len(orders), "per_category": {c: cats[c] for c in CATEGORIES},
            "per_status": {s: stats[s] for s in STATUSES}, "delayed_pct": round(delayed_pct, 2)}


ORDERS = generate_orders()
ORDERS_BY_ID = {o["record_id"]: o for o in ORDERS}

if __name__ == "__main__":
    report = validate_orders(ORDERS)
    print(f"seed={SEED}  orders={report['total']}")
    print("count per category:", report["per_category"])
    print("count per status  :", report["per_status"])
    print(f"delayed_shipment=True: {report['delayed_pct']}%  (required band 10-30%)")
    print("first 3 records   :", *ORDERS[:3], sep="\n  ")
