"""Input guardrails (PII masking, prompt-injection) and output groundedness check (Task 10)."""
import re

_CARD_FULL = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_CARD_A = re.compile(r"(?i)\b(?:ending|ends|last\s*(?:4|four)(?:\s*digits?)?)(?:\s*(?:in|with|is|are|of))*\s*[:\-]?\s*(\d{4})(?!\d)")
_CARD_B = re.compile(r"(?i)\bcard\s*(?:no\.?|number|#)?\s*(?:is\s*)?[:\-]?\s*(\d{4})(?!\d)")
_CARD_MASKED = re.compile(r"(?i)\b[x*\u2022]{4,}[ -]?\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def _sub_group(pattern, repl, text):
    def f(m):
        s, e = m.span(1)
        return m.group(0)[: s - m.start()] + repl + m.group(0)[e - m.start():]
    return pattern.subn(f, text)


def mask_pii(text: str):
    """-> (masked_text, sorted list of PII types found). Phone and card last-4 are fixed-format."""
    found = set()
    text, n = _CARD_FULL.subn("[CARD_NUMBER]", text)
    if n: found.add("card_number")
    for pat in (_CARD_A, _CARD_B):
        text, n = _sub_group(pat, "[CARD_LAST4]", text)
        if n: found.add("card_last4")
    text, n = _CARD_MASKED.subn("[CARD_LAST4]", text)
    if n: found.add("card_last4")
    text, n = _PHONE.subn("[PHONE]", text)
    if n: found.add("phone")
    return text, sorted(found)


_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)",
    r"\byou are now\b", r"\bact as (an? )?(unrestricted|jailbroken|dan)\b", r"developer mode",
    r"jailbreak", r"bypass (the )?(guardrails?|safety|filters?)", r"forget (everything|all) (you|your)",
)]


def detect_injection(text: str):
    """-> the matched pattern string, or None."""
    for p in _INJECTION:
        if p.search(text):
            return p.pattern
    return None


_STOP = {"the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "your", "you", "can", "will",
         "has", "have", "not", "but", "any", "all", "our", "its", "their", "than", "then", "also", "into", "such"}


def content_tokens(text: str) -> set:
    toks = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t.isdigit() or (len(t) > 2 and t not in _STOP):
            toks.add(t)
    return toks


def groundedness_check(answer: str, contexts: list, top_score, threshold: float, min_overlap: float = 0.7):
    """Output-side guardrail -> (ok, reason). Refuse if retrieval is weak OR the answer has unsupported content."""
    if not contexts or top_score is None or top_score < threshold:
        return False, f"top similarity {top_score} is below calibrated threshold {threshold}"
    a, c = content_tokens(answer), set().union(*(content_tokens(x) for x in contexts))
    if not a:
        return False, "empty answer"
    overlap = len(a & c) / len(a)
    if overlap < min_overlap:
        return False, f"only {overlap:.0%} of answer content appears in retrieved context (min {min_overlap:.0%})"
    return True, "supported by retrieved context"
