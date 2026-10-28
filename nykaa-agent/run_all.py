"""Runs every demo in order and saves each transcript to transcripts/ (all under MOCK_LLM, no API keys).
Usage: python run_all.py"""
import os
import re
import subprocess
import sys

from config import ARTIFACTS, ROOT, TRANSCRIPTS

STEPS = [
    ("01_dataset", ["dataset.py"]),
    ("02_rag_eval_calibration", ["-m", "evaluation.rag_eval"]),
    ("03_escalation_tool", ["-m", "agent.tools"]),
    ("04_routing", ["-m", "scripts.demo_agent", "routing"]),
    ("05_memory", ["-m", "scripts.demo_agent", "memory"]),
    ("06_guardrails_schema", ["-m", "scripts.demo_agent", "guardrails"]),
    ("07_fastapi_logging", ["-m", "scripts.demo_api"]),
    ("08_rag_triad", ["-m", "evaluation.triad"]),
    ("09_mcp", ["-m", "mcp_layer.demo"]),
    ("10_checkpointing", ["-m", "resilience.checkpoint_demo"]),
    ("11_retry_timeouts", ["-m", "resilience.retry_timeout_demo"]),
]

if __name__ == "__main__":
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONPATH": str(ROOT)}
    failed = []
    for name, args in STEPS:
        print(f"==> {name}", flush=True)
        p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
        (TRANSCRIPTS / f"{name}.txt").write_text(p.stdout + (f"\n--- STDERR ---\n{p.stderr}" if p.returncode else ""))
        print("    OK" if p.returncode == 0 else f"    FAILED (see transcripts/{name}.txt)", flush=True)
        if p.returncode:
            failed.append(name)
    results = "\n".join(f.read_text() for f in sorted(ARTIFACTS.glob("results_*.md")))
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", lambda m: f"{m.group(1)}\n{results}\n{m.group(2)}", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print("\nFAILED STEPS:" if failed else "\nAll steps passed.", failed or "")
    sys.exit(1 if failed else 0)
