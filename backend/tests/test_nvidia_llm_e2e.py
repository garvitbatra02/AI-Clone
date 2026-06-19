"""
End-to-end tests for NVIDIA NIM LLM provider.

Tests:
  1. Direct chat       — Nemotron Nano, Meta Llama, Mistral
  2. Streaming          — Nemotron Nano, Meta Llama
  3. Async chat         — Nemotron Nano, Meta Llama
  4. Async streaming    — Meta Llama
  5. Reasoning model    — Nemotron Nano (reasoning_content fallback)
  6. Factory auto-detect from model name
  7. Multi-turn conversation
  8. System prompt adherence

Run:
    PYTHONPATH=. python tests/test_nvidia_llm_e2e.py

Requires:
    NVIDIA_API_KEYS (or NVIDIA_API_KEY) set in .env or environment.
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


def _has_nvidia_keys() -> bool:
    """Check if NVIDIA API keys are available."""
    return bool(
        os.environ.get("NVIDIA_API_KEYS", "").strip()
        or os.environ.get("NVIDIA_API_KEY", "").strip()
    )


# ── Pre-flight ───────────────────────────────────────────────────

if not _has_nvidia_keys():
    print("❌ NVIDIA_API_KEYS not set. Skipping NVIDIA tests.")
    sys.exit(0)


# ── Create LLM instances ─────────────────────────────────────────

nemotron_nano = LLMFactory.create(
    provider=LLMProvider.NVIDIA,
    model="nvidia/nemotron-3-nano-30b-a3b",
    max_tokens=100,
)

llama = LLMFactory.create(
    provider=LLMProvider.NVIDIA,
    model="meta/llama-3.3-70b-instruct",
    max_tokens=100,
)


# ──────────────────────────────────────────────────────────────
#  1. Direct Chat
# ──────────────────────────────────────────────────────────────
sep("1. Direct Chat — Nemotron Nano, Meta Llama")

for name, llm, question in [
    ("Nemotron Nano", nemotron_nano, "What is the capital of Japan?"),
    ("Meta Llama 3.3", llama, "What is 7 * 8?"),
]:
    try:
        s = make_session("Be brief, one sentence max.", question)
        r = llm.chat(s)
        check(
            f"{name} chat",
            bool(r.content),
            f"tokens={r.total_tokens}, provider={r.provider}, response={r.content[:80]}",
        )
    except Exception as e:
        check(f"{name} chat", False, str(e))


# ──────────────────────────────────────────────────────────────
#  2. Streaming
# ──────────────────────────────────────────────────────────────
sep("2. Streaming — Nemotron Nano, Meta Llama")

for name, llm, question in [
    ("Nemotron Nano", nemotron_nano, "Count from 1 to 5."),
    ("Meta Llama 3.3", llama, "List 3 colors."),
]:
    try:
        s = make_session("Be brief.", question)
        chunks = []
        for chunk in llm.chat_stream(s):
            chunks.append(chunk)
        full = "".join(chunks)
        check(f"{name} stream", len(chunks) > 0, f"{len(chunks)} chunks, text={full[:80]}")
    except Exception as e:
        check(f"{name} stream", False, str(e))


# ──────────────────────────────────────────────────────────────
#  3 & 4. Async Chat + Async Streaming
# ──────────────────────────────────────────────────────────────
sep("3. Async Chat — Nemotron Nano, Meta Llama")


async def test_async():
    s = make_session("One sentence max.", "What is gravity?")

    for name, llm in [("Nemotron Nano", nemotron_nano), ("Meta Llama 3.3", llama)]:
        try:
            r = await llm.chat_async(s)
            check(f"{name} async chat", bool(r.content), r.content[:80])
        except Exception as e:
            check(f"{name} async chat", False, str(e))

    # Async streaming
    print()
    sep("4. Async Streaming — Meta Llama")
    try:
        s2 = make_session("Be brief.", "Say hello in 3 languages.")
        chunks = []
        async for chunk in llama.chat_stream_async(s2):
            chunks.append(chunk)
        full = "".join(chunks)
        check("Meta Llama async stream", len(chunks) > 0, f"{len(chunks)} chunks, text={full[:80]}")
    except Exception as e:
        check("Meta Llama async stream", False, str(e))


asyncio.run(test_async())


# ──────────────────────────────────────────────────────────────
#  5. Reasoning Model — content fallback
# ──────────────────────────────────────────────────────────────
sep("5. Reasoning Model — Nemotron Nano (reasoning_content)")

try:
    # Nemotron reasoning models sometimes return content in reasoning_content
    # Our implementation should handle this transparently
    s = make_session("Answer concisely.", "What is 15 * 4?")
    r = nemotron_nano.chat(s)
    check(
        "Reasoning model response not empty",
        bool(r.content.strip()),
        f"content={r.content[:100]}",
    )
    check(
        "Provider is nvidia",
        r.provider == "nvidia",
        f"provider={r.provider}",
    )
    check(
        "Usage stats present",
        r.usage is not None and r.usage.get("total_tokens", 0) > 0,
        f"usage={r.usage}",
    )
except Exception as e:
    check("Reasoning model", False, str(e))


# ──────────────────────────────────────────────────────────────
#  6. Factory auto-detect from model name
# ──────────────────────────────────────────────────────────────
sep("6. Factory — Auto-detect NVIDIA Provider from Model")

nvidia_test_models = [
    "nvidia/nemotron-3-nano-30b-a3b",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-medium-3.5-128b",
    "deepseek-ai/deepseek-v4-flash",
]

for model_name in nvidia_test_models:
    try:
        detected = get_provider_for_model(model_name)
        check(
            f"Registry: '{model_name}'",
            detected == LLMProvider.NVIDIA,
            f"detected={detected}",
        )
    except Exception as e:
        check(f"Registry: '{model_name}'", False, str(e))

# Test from_model creates working instance
try:
    auto_llm = LLMFactory.from_model(model="nvidia/nemotron-3-nano-30b-a3b", max_tokens=50)
    check(
        "from_model() creates NVIDIA instance",
        auto_llm.provider == LLMProvider.NVIDIA,
        f"provider={auto_llm.provider.value}",
    )
    s = make_session("One word.", "What color is the sky?")
    r = auto_llm.chat(s)
    check("  → chat works", bool(r.content), r.content[:60])
except Exception as e:
    check("from_model()", False, str(e))


# ──────────────────────────────────────────────────────────────
#  7. Multi-turn Conversation
# ──────────────────────────────────────────────────────────────
sep("7. Multi-turn Conversation (Meta Llama)")

conv = ChatSession()
conv.add_system_prompt("You are a math tutor. Be brief.")
questions = ["What is 2+2?", "Multiply that by 3.", "Subtract 5."]

for q in questions:
    try:
        conv.add_user_message(q)
        r = llama.chat(conv)
        check(f"Q: {q}", bool(r.content), f"A: {r.content.strip()[:80]}")
        conv.add_assistant_message(r.content)
    except Exception as e:
        check(f"Q: {q}", False, str(e))


# ──────────────────────────────────────────────────────────────
#  8. System Prompt Adherence
# ──────────────────────────────────────────────────────────────
sep("8. System Prompt Adherence")

try:
    s = make_session(
        "You must respond with ONLY a JSON object. No other text.",
        "Give me a JSON with keys 'name' and 'age' for a person named Alice who is 30.",
    )
    r = llama.chat(s)
    # Check that response contains JSON-like structure
    has_json = "{" in r.content and "}" in r.content
    check("JSON format adherence", has_json, f"response={r.content[:100]}")
except Exception as e:
    check("System prompt adherence", False, str(e))


# ──────────────────────────────────────────────────────────────
#  9. Model Registry completeness
# ──────────────────────────────────────────────────────────────
sep("9. Model Registry — NVIDIA Coverage")

all_nvidia_models = get_all_models_for_provider(LLMProvider.NVIDIA)
check(
    "NVIDIA models registered",
    len(all_nvidia_models) >= 30,
    f"count={len(all_nvidia_models)}",
)

# Verify all use publisher/model format
slash_models = [m for m in all_nvidia_models if "/" in m]
check(
    "All use publisher/model format",
    len(slash_models) == len(all_nvidia_models),
    f"{len(slash_models)}/{len(all_nvidia_models)} have slash",
)


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
    print("\n  ✅ ALL NVIDIA LLM CHECKS PASSED")
    sys.exit(0)
