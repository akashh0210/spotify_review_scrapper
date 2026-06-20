"""Smoke test — confirms GEMINI_API_KEY works with the google-genai SDK.

Run:  python src/test_gemini.py
Expected output: raw JSON response from Gemini and "Key works!" message.
"""

import os
from dotenv import load_dotenv

load_dotenv(".env.local") or load_dotenv()

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: GEMINI_API_KEY not set in .env.local")

print(f"Testing key: {api_key[:8]}...{api_key[-4:]}")
print("Sending single test call to gemini-2.0-flash...\n")

client = genai.Client(api_key=api_key)

resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents='Return exactly this JSON and nothing else: {"status": "ok", "model": "gemini-2.0-flash"}',
    config={"temperature": 0, "response_mime_type": "application/json"},
)

print("Raw response:")
print(resp.text)
print()
if resp.text and "ok" in resp.text:
    print("Key works -- ready for retag_errors.py")
else:
    print("Unexpected response — check key or quota.")
