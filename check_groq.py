"""Smoke test: confirm Groq + Llama 3.3 70B is reachable from this machine.

Safe to delete after Stage 1 completes successfully.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not found. Check that .env exists in the project root "
        "and contains: GROQ_API_KEY=gsk_..."
    )
if not api_key.startswith("gsk_"):
    print(f"WARNING: key doesn't start with 'gsk_' - is it really a Groq key?")

print(f"API key found: {api_key[:8]}...{api_key[-4:]}  (length: {len(api_key)})")
print("Sending one test message to Llama 3.3 70B on Groq ...\n")

client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "In exactly one sentence, what is Apple Inc.?"},
    ],
    max_tokens=80,
    temperature=0.2,
)

print("Response:")
print(response.choices[0].message.content)
print()
print(f"Model:  {response.model}")
print(f"Usage:  prompt={response.usage.prompt_tokens} "
      f"completion={response.usage.completion_tokens} "
      f"total={response.usage.total_tokens}")
print("\nGroq is reachable. Stage 1 complete.")
