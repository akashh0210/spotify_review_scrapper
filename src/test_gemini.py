"""Probe which Gemini models have free-tier quota available.

Tests each model with a single tiny call and prints a summary table:
  model | status | limit/error detail

Run:  python src/test_gemini.py
"""

import os
import time
from dotenv import load_dotenv

load_dotenv(".env.local") or load_dotenv()

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: GEMINI_API_KEY not set in .env.local")

print(f"Key: {api_key[:8]}...{api_key[-4:]}")
print()

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-2.5-flash-lite",   # primary (quota confirmed)
    "gemini-2.5-flash",        # fallback (quota confirmed)
    "gemini-2.0-flash",        # quota-zeroed on this key
    "gemini-2.0-flash-lite",   # quota-zeroed on this key
]

PROMPT = 'Return exactly: {"status": "ok"}'

results = []

for model in MODELS:
    print(f"Testing {model} ...", flush=True)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=PROMPT,
            config={"temperature": 0, "response_mime_type": "application/json"},
        )
        results.append((model, "OK", resp.text.strip()[:60]))
    except Exception as exc:
        msg = str(exc)
        # Extract the key diagnostic fields from the error
        code = "?"
        limit_val = "?"
        if "429" in msg:
            code = "429 RESOURCE_EXHAUSTED"
        elif "404" in msg:
            code = "404 NOT_FOUND"
        elif "400" in msg:
            code = "400 BAD_REQUEST"
        elif "403" in msg:
            code = "403 FORBIDDEN"
        else:
            code = msg[:40]

        # Pull out "limit: N" if present
        import re
        m = re.search(r"limit:\s*(\d+)", msg)
        limit_val = m.group(0) if m else "—"

        # Pull out quotaId if present
        qm = re.search(r'"quotaId":\s*"([^"]+)"', msg)
        quota_id = qm.group(1) if qm else "—"

        results.append((model, code, f"{limit_val} | quotaId={quota_id}"))

    time.sleep(2)  # brief pause between calls

# ── summary table ──────────────────────────────────────────────────────────────
print()
print(f"{'Model':<28} {'Status':<28} Detail")
print("-" * 90)
for model, status, detail in results:
    print(f"{model:<28} {status:<28} {detail}")
print()

working = [m for m, s, _ in results if s == "OK"]
if working:
    print(f"Working models: {working}")
else:
    print("No models succeeded — all quota exhausted or unavailable on this project.")
    print("Options: wait for daily quota reset, enable billing, or use Groq only.")
