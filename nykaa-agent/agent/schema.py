"""Structured-output contract (Task 9): every agent response is validated against this JSON Schema."""
RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentResponse", "type": "object", "additionalProperties": False,
    "required": ["trace_id", "session_id", "query", "intent", "answer", "sources", "grounded",
                 "refused", "guardrails", "escalation", "top_score"],
    "properties": {
        "trace_id": {"type": "string", "minLength": 1},
        "session_id": {"type": "string", "minLength": 1},
        "query": {"type": "string"},
        "intent": {"enum": ["policy", "order_status", "blocked"]},
        "answer": {"type": "string", "minLength": 1},
        "sources": {"type": "array", "items": {"type": "string"}},
        "grounded": {"type": "boolean"},
        "refused": {"type": "boolean"},
        "guardrails": {"type": "array", "items": {"type": "string"}},
        "escalation": {"type": ["object", "null"], "required": ["score", "threshold", "recommended"],
                       "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1},
                                      "threshold": {"type": "number"}, "recommended": {"type": "boolean"}}},
        "top_score": {"type": ["number", "null"]},
    },
}


def validate_response(resp: dict) -> None:
    import jsonschema
    jsonschema.validate(instance=resp, schema=RESPONSE_SCHEMA)
