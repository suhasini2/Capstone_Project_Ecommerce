"""LLM access. MOCK_LLM=1 (default): everything is deterministic and keyless.
MOCK_LLM=0 + GROQ_API_KEY: optional real model (untested by the graded run)."""
import os
from config import MOCK_LLM


def complete(prompt: str) -> str:
    if MOCK_LLM:
        raise RuntimeError("complete() must not be called under MOCK_LLM")
    from groq import Groq  # pip install groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                                       messages=[{"role": "user", "content": prompt}], temperature=0)
    return r.choices[0].message.content.strip()
