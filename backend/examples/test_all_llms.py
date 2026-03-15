"""
Comprehensive LLM smoke-test across every provider and call pattern.

Tests:
  1. Direct chat       — Groq, Cerebras, Cohere
  2. Streaming          — Groq, Cerebras, Cohere
  3. Async chat         — Groq, Cerebras, Cohere
  4. Async streaming    — Groq
  5. ChatService rotation + fallback
  6. Factory auto-detect from model name
  7. Multi-turn conversation (Cohere)

Run:
    PYTHONPATH=. python examples/test_all_llms.py
"""

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from ChatService.Chat import (
    ChatSession,
    LLMFactory,
    LLMProvider,
    get_chat_service,
    chat_inference,
    AllProvidersFailedError,
    AllKeysFailedError,
)


def sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_session(system: str, user: str) -> ChatSession:
    s = ChatSession()
    s.add_system_prompt(system)
    s.add_user_message(user)
    return s


# ──────────────────────────────────────────────────────────────
#  Create one instance of each provider
# ──────────────────────────────────────────────────────────────
groq = LLMFactory.create(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
cerebras = LLMFactory.create(provider=LLMProvider.CEREBRAS, model="llama3.1-8b")
cohere = LLMFactory.create(provider=LLMProvider.COHERE, model="command-r7b-12-2024")

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


# ──────────────────────────────────────────────────────────────
#  1. Direct Chat
# ──────────────────────────────────────────────────────────────
sep("1. Direct Chat — Groq, Cerebras, Cohere")

for name, llm, question in [
    ("Groq", groq, "What is the capital of France?"),
    ("Cerebras", cerebras, "What is 7 * 8?"),
    ("Cohere", cohere, "What is the speed of light?"),
]:
    try:
        s = make_session("Be brief, one sentence max.", question)
        r = llm.chat(s)
        check(f"{name} chat", bool(r.content), f"tokens={r.total_tokens}, response={r.content[:80]}")
    except Exception as e:
        check(f"{name} chat", False, str(e))


# ──────────────────────────────────────────────────────────────
#  2. Streaming
# ──────────────────────────────────────────────────────────────
sep("2. Streaming — Groq, Cerebras, Cohere")

for name, llm, question in [
    ("Groq", groq, "Count from 1 to 5."),
    ("Cerebras", cerebras, "List 3 colors."),
    ("Cohere", cohere, "Name 3 planets."),
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
sep("3. Async Chat — Groq, Cerebras, Cohere")


async def test_async():
    s = make_session("One sentence max.", "What is gravity?")

    for name, llm in [("Groq", groq), ("Cerebras", cerebras), ("Cohere", cohere)]:
        try:
            r = await llm.chat_async(s)
            check(f"{name} async chat", bool(r.content), r.content[:80])
        except Exception as e:
            check(f"{name} async chat", False, str(e))

    # Async streaming (Groq)
    print()
    sep("4. Async Streaming — Groq")
    try:
        s2 = make_session("Be brief.", "Say hello in 3 languages.")
        chunks = []
        async for chunk in groq.chat_stream_async(s2):
            chunks.append(chunk)
        full = "".join(chunks)
        check("Groq async stream", len(chunks) > 0, f"{len(chunks)} chunks, text={full[:80]}")
    except Exception as e:
        check("Groq async stream", False, str(e))


asyncio.run(test_async())


# ──────────────────────────────────────────────────────────────
#  5. ChatService — Rotation + Fallback
# ──────────────────────────────────────────────────────────────
sep("5. ChatService — Rotation + Fallback")

service = get_chat_service()
print(f"  Providers: {service.get_active_providers()}")
print(f"  Count: {service.get_provider_count()}")

providers_used = []
for i in range(4):
    try:
        s = make_session("One word answer.", f"What is {i + 1}+{i + 1}?")
        provider_name = service.get_current_provider().provider.value
        r = service.chat(s, fallback=True)
        providers_used.append(provider_name)
        check(
            f"Turn {i + 1}",
            bool(r.content),
            f"provider={provider_name}, answer={r.content.strip()[:50]}",
        )
    except Exception as e:
        check(f"Turn {i + 1}", False, str(e))

# Verify rotation happened (at least 2 different providers used)
unique_providers = set(providers_used)
check(
    "Provider rotation",
    len(unique_providers) >= 2,
    f"used {unique_providers}",
)


# ──────────────────────────────────────────────────────────────
#  6. Factory auto-detect from model name
# ──────────────────────────────────────────────────────────────
sep("6. Factory — Auto-detect Provider from Model")

for model_name, expected_provider in [
    ("llama-3.3-70b-versatile", "groq"),
    ("llama3.1-8b", "cerebras"),
    ("command-r7b-12-2024", "cohere"),
]:
    try:
        auto_llm = LLMFactory.from_model(model=model_name)
        check(
            f"from_model('{model_name}')",
            auto_llm.provider.value == expected_provider,
            f"detected={auto_llm.provider.value}",
        )
        s = make_session("One sentence.", "What is Python?")
        r = auto_llm.chat(s)
        check(
            f"  → chat works",
            bool(r.content),
            r.content[:60],
        )
    except Exception as e:
        check(f"from_model('{model_name}')", False, str(e))


# ──────────────────────────────────────────────────────────────
#  7. Multi-turn conversation
# ──────────────────────────────────────────────────────────────
sep("7. Multi-turn Conversation (Cohere)")

conv = ChatSession()
conv.add_system_prompt("You are a math tutor. Be brief.")
questions = ["What is 2+2?", "Multiply that by 3.", "Subtract 5."]

for q in questions:
    try:
        conv.add_user_message(q)
        r = cohere.chat(conv)
        check(f"Q: {q}", bool(r.content), f"A: {r.content.strip()[:80]}")
        conv.add_assistant_message(r.content)
    except Exception as e:
        check(f"Q: {q}", False, str(e))


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
    print("\n  ✅ ALL LLM CHECKS PASSED")
    sys.exit(0)
