"""Rule-based intent router (deterministic under MOCK_LLM). Returns (intent, record_id)."""
import re

ORDER_ID = re.compile(r"\bNYK-\d{4}\b", re.I)
FOLLOWUP = re.compile(r"\b(it|that order|this order|the same order|same one)\b", re.I)
ORDER_WORDS = re.compile(r"\b(status|track\w*|where is|escalat\w*|delay\w*|shipped|delivered|arriv\w*|value|price)\b", re.I)
GENERIC_ORDER = re.compile(r"\b(order status|track my order|where is my order|status of my order|check my order)\b", re.I)
POLICY_LIKE = re.compile(r"\b(can i|how (do|does|long|many)|what (is|are|happens)|policy|eligible|allowed)\b", re.I)


def classify_intent(query: str, last_record_id=None):
    m = ORDER_ID.search(query)
    if m:
        return "order_status", m.group(0).upper()
    followup = FOLLOWUP.search(query) and ORDER_WORDS.search(query) and not POLICY_LIKE.search(query)
    if followup:
        return "order_status", last_record_id  # None -> the order node asks for the ID
    if GENERIC_ORDER.search(query):
        return "order_status", last_record_id
    return "policy", None
