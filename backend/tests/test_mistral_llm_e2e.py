"""
End-to-end tests for Mistral AI LLM provider.

Tests:
  1. Direct chat       — mistral-small, mistral-medium, mistral-large
  2. Streaming          — mistral-small
  3. Async chat         — mistral-small
  4. Async streaming    — mistral-small
  5. Code model         — codestral-latest
  6. Factory auto-detect from model name
  7. Multi-turn conversation
  8. Key rotation (3 keys available)
  9. Model registry completeness

Run:
    PYTHONPATH=. python tests/test_mistral_llm_e2e.py

Requires:
    MISTRAL_API_KEYS set in .env or environment.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Fix aiohttp SSL cert verification on macOS
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

from ChatService.Chat import (
    ChatSession,
    LLMFactory,
    LLMProvider,
    AllKeysFailedError,
)
from ChatService.Chat.llm.model_registry import get_provider_for_model, get_all_models_for_provider


# ── Helpers ──────────────────────────────────────────────────────

def sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_session(system: str, user: str) -> ChatSession:
    s = ChatSession()
    s.add_system_prompt(system)
    s.add_user_message(user)
    return s


passed = 0
failed = 0


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))


def _has_mistral_keys() -> bool:
    return bool(
        os.environ.get("MISTRAL_API_KEYS", "").strip()
        or os.environ.get("MISTRAL_API_KEY", "").strip()
    )


# ── Pre-flight ───────────────────────────────────────────────────

if not _has_mistral_keys():
    print("❌ MISTRAL_API_KEYS not set. Skipping Mistral tests.")
    sys.exit(0)


# ── Create LLM instances ─────────────────────────────────────────

small = LLMFactory.create(provider=LLMProvider.MISTRAL, model="mistral-small-latest", max_tokens=100)
medium = LLMFactory.create(provider=LLMProvider.MISTRAL, model="mistral-medium-latest", max_tokens=100)


# ──────────────────────────────────────────────────────────────
#  1. Direct Chat
# ──────────────────────────────────────────────────────────────
sep("1. Direct Chat — Small, Medium, Large")

for name, model_id in [
    ("Mistral Small", "mistral-small-latest"),
    ("Mistral Medium", "mistral-medium-latest"),
    ("Mistral Large", "mistral-large-latest"),
]:
    try:
        llm = LLMFactory.create(provider=LLMProvider.MISTRAL, model=model_id, max_tokens=50)
        s = make_session("Be brief, one sentence max.", "What is the capital of Japan?")
        r = llm.chat(s)
        check(
            f"{name} chat",
            bool(r.content),
            f"tokens={r.total_tokens}, response={r.content[:80]}",
        )
    except Exception as e:
        check(f"{name} chat", False, str(e))


# ──────────────────────────────────────────────────────────────
#  2. Streaming
# ──────────────────────────────────────────────────────────────
sep("2. Streaming — Mistral Small")

try:
    s = make_session("Be brief.", "List 3 fruits.")
    chunks = list(small.chat_stream(s))
    full = "".join(chunks)
    check("Mistral Small stream", len(chunks) > 0, f"{len(chunks)} chunks, text={full[:80]}")
except Exception as e:
    check("Mistral Small stream", False, str(e))


# ──────────────────────────────────────────────────────────────
#  3 & 4. Async Chat + Async Streaming
# ──────────────────────────────────────────────────────────────
sep("3. Async Chat — Mistral Small")


async def test_async():
    try:
        s = make_session("One sentence.", "What is gravity?")
        r = await small.chat_async(s)
        check("Mistral Small async chat", bool(r.content), r.content[:80])
    except Exception as e:
        check("Mistral Small async chat", False, str(e))

    print()
    sep("4. Async Streaming — Mistral Small")
    try:
        s2 = make_session("Be brief.", "Say hello in 3 languages.")
        chunks = []
        async for chunk in small.chat_stream_async(s2):
            chunks.append(chunk)
        full = "".join(chunks)
        check("Mistral Small async stream", len(chunks) > 0, f"{len(chunks)} chunks, text={full[:80]}")
    except Exception as e:
        check("Mistral Small async stream", False, str(e))


asyncio.run(test_async())


# ──────────────────────────────────────────────────────────────
#  5. Code Model — Codestral
# ──────────────────────────────────────────────────────────────
sep("5. Code Model — Codestral")

try:
    code_llm = LLMFactory.create(provider=LLMProvider.MISTRAL, model="codestral-latest", max_tokens=100)
    s = make_session("Respond with code only.", "Write a Python function to check if a number is prime.")
    r = code_llm.chat(s)
    has_code = "def" in r.content or "return" in r.content
    check("Codestral code generation", has_code, f"response={r.content[:100]}")
except Exception as e:
    check("Codestral code generation", False, str(e))


# ──────────────────────────────────────────────────────────────
#  6. Factory auto-detect from model name
# ──────────────────────────────────────────────────────────────
sep("6. Factory — Auto-detect Mistral Provider from Model")

test_models = [
    "mistral-small-latest",
    "mistral-large-latest",
    "codestral-latest",
    "magistral-medium-latest",
    "ministral-8b-latest",
]

for model_name in test_models:
    detected = get_provider_for_model(model_name)
    check(f"Registry: '{model_name}'", detected == LLMProvider.MISTRAL, f"detected={detected}")

try:
    auto_llm = LLMFactory.from_model(model="mistral-small-latest", max_tokens=30)
    check("from_model() creates Mistral instance", auto_llm.provider == LLMProvider.MISTRAL)
    s = make_session("One word.", "What color is grass?")
    r = auto_llm.chat(s)
    check("  → chat works", bool(r.content), r.content[:60])
except Exception as e:
    check("from_model()", False, str(e))


# ──────────────────────────────────────────────────────────────
#  7. Multi-turn Conversation
# ──────────────────────────────────────────────────────────────
sep("7. Multi-turn Conversation")

conv = ChatSession()
conv.add_system_prompt("You are a math tutor. Be brief.")
questions = ["What is 3+3?", "Multiply that by 2.", "Subtract 4."]

for q in questions:
    try:
        conv.add_user_message(q)
        r = small.chat(conv)
        check(f"Q: {q}", bool(r.content), f"A: {r.content.strip()[:80]}")
        conv.add_assistant_message(r.content)
    except Exception as e:
        check(f"Q: {q}", False, str(e))


# ──────────────────────────────────────────────────────────────
#  8. Key Rotation
# ──────────────────────────────────────────────────────────────
sep("8. Key Rotation — 3 keys configured")

keys = os.environ.get("MISTRAL_API_KEYS", "").split(",")
check("Multiple keys loaded", len(keys) >= 3, f"count={len(keys)}")


# ──────────────────────────────────────────────────────────────
#  9. Model Registry
# ──────────────────────────────────────────────────────────────
sep("9. Model Registry — Mistral Coverage")

all_models = get_all_models_for_provider(LLMProvider.MISTRAL)
check("Mistral models registered", len(all_models) >= 25, f"count={len(all_models)}")


# ──────────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────────
sep("SUMMARY")
total = passed + failed
print(f"  Passed: {passed}/{total}")
print(f"  Failed: {failed}/{total}")

if failed:
    print("\n  ⚠️  SOME CHECKS FAILED — see above")
    sys.exit(1)
else:
    print("\n  ✅ ALL MISTRAL LLM CHECKS PASSED")
    sys.exit(0)
