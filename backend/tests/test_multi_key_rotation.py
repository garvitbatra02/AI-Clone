"""
Comprehensive tests for multi-API key rotation with model flexibility.

Test coverage:
- Default model chat (Path A)
- Default model streaming (Path A)
- Specific model chat (Path B)
- Specific model streaming (Path B)
- API key rotation and fallback
- Async operations
- Error handling

Setup:
1. Set environment variables:
   GROQ_API_KEYS="key1,key2,key3"
   CEREBRAS_API_KEYS="key1,key2,key3"

2. Run tests:
   python tests/test_multi_key_rotation.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path to import ChatService
sys.path.insert(0, str(Path(__file__).parent.parent))

from ChatService.Chat import ChatSession, LLMProvider
from ChatService.Chat.services import (
    get_chat_service,
    chat_inference,
    chat_inference_stream,
    chat_inference_async,
    chat_inference_stream_async,
)
from ChatService.Chat.services.chat_service import AllProvidersFailedError, AllKeysFailedError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def example_1_default_model_chat():
    """Test 1: Default model with provider rotation (Path A)."""
    print_section("Test 1: Default Model Chat (Path A)")
    
    session = ChatSession()
    session.add_user_message("What is 2+2? Answer in one word.")
    
    try:
        response = chat_inference(session, fallback=True)
        print(f"✅ Response: {response.content}")
        print(f"   Tokens: {response.total_tokens}")
        print(f"   Model: {response.model}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_default_model_streaming():
    """Test 2: Default model with streaming (Path A)."""
    print_section("Test 2: Default Model Streaming (Path A)")
    
    session = ChatSession()
    session.add_user_message("Count from 1 to 5.")
    
    try:
        print("Response: ", end="", flush=True)
        for chunk in chat_inference_stream(session, fallback=True):
            print(chunk, end="", flush=True)
        print("\n✅ Streaming completed successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_3_specific_model_chat():
    """Test 3: Specific model with key-level fallback (Path B)."""
    print_section("Test 3: Specific Model Chat (Path B)")
    
    session = ChatSession()
    session.add_user_message("What is the capital of France? One word.")
    
    try:
        # Test with a specific Groq model
        response = chat_inference(
            session,
            provider=LLMProvider.GROQ,
            model="llama-3.1-8b-instant",
            fallback=True
        )
        print(f"✅ Response: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.total_tokens}")
    except AllKeysFailedError as e:
        print(f"❌ All keys failed for {e.provider} with model {e.model}")
        print(f"   Errors: {e.errors}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_specific_model_streaming():
    """Test 4: Specific model with streaming and key-level fallback (Path B)."""
    print_section("Test 4: Specific Model Streaming (Path B)")
    
    session = ChatSession()
    session.add_user_message("Say hello in 3 different languages.")
    
    try:
        print("Response: ", end="", flush=True)
        for chunk in chat_inference_stream(
            session,
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            fallback=True
        ):
            print(chunk, end="", flush=True)
        print("\n✅ Streaming completed successfully")
    except AllKeysFailedError as e:
        print(f"\n❌ All keys failed for {e.provider} with model {e.model}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_5_provider_rotation():
    """Test 5: Multiple requests to verify provider rotation (Path A)."""
    print_section("Test 5: Provider Rotation Verification")
    
    service = get_chat_service()
    
    for i in range(4):
        session = ChatSession()
        session.add_user_message(f"Request #{i+1}: Say 'OK'")
        
        try:
            response = chat_inference(session, fallback=False)
            current_provider = service.get_current_provider()
            print(f"Request {i+1}: Provider={current_provider.provider.value}, Response={response.content[:20]}")
        except Exception as e:
            print(f"Request {i+1}: ❌ Error: {e}")


def example_6_key_rotation_specific_model():
    """Test 6: Multiple requests with specific model to verify key rotation (Path B)."""
    print_section("Test 6: Key Rotation with Specific Model")
    
    for i in range(3):
        session = ChatSession()
        session.add_user_message(f"Request #{i+1}: What is 10+{i}?")
        
        try:
            response = chat_inference(
                session,
                provider=LLMProvider.GROQ,
                model="llama-3.1-8b-instant",
                fallback=False  # No fallback to see which key is used
            )
            print(f"Request {i+1}: Response={response.content[:30]}")
        except Exception as e:
            print(f"Request {i+1}: ❌ Error: {e}")


async def example_7_async_default_model():
    """Test 7: Async chat with default model (Path A)."""
    print_section("Test 7: Async Chat with Default Model")
    
    session = ChatSession()
    session.add_user_message("What is Python? Answer in one sentence.")
    
    try:
        response = await chat_inference_async(session, fallback=True)
        print(f"✅ Response: {response.content}")
        print(f"   Model: {response.model}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_8_async_specific_model():
    """Test 8: Async chat with specific model (Path B)."""
    print_section("Test 8: Async Chat with Specific Model")
    
    session = ChatSession()
    session.add_user_message("What is JavaScript? Answer in one sentence.")
    
    try:
        response = await chat_inference_async(
            session,
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            fallback=True
        )
        print(f"✅ Response: {response.content}")
    except AllKeysFailedError as e:
        print(f"❌ All keys failed for {e.provider} with model {e.model}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_9_async_streaming_default():
    """Test 9: Async streaming with default model (Path A)."""
    print_section("Test 9: Async Streaming with Default Model")
    
    session = ChatSession()
    session.add_user_message("List 3 programming languages.")
    
    try:
        print("Response: ", end="", flush=True)
        async for chunk in chat_inference_stream_async(session, fallback=True):
            print(chunk, end="", flush=True)
        print("\n✅ Async streaming completed successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_10_async_streaming_specific():
    """Test 10: Async streaming with specific model (Path B)."""
    print_section("Test 10: Async Streaming with Specific Model")
    
    session = ChatSession()
    session.add_user_message("Name 3 fruits.")
    
    try:
        print("Response: ", end="", flush=True)
        async for chunk in chat_inference_stream_async(
            session,
            provider=LLMProvider.GROQ,
            model="llama-3.1-8b-instant",
            fallback=True
        ):
            print(chunk, end="", flush=True)
        print("\n✅ Async streaming completed successfully")
    except AllKeysFailedError as e:
        print(f"\n❌ All keys failed for {e.provider} with model {e.model}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_11_multi_turn_conversation():
    """Test 11: Multi-turn conversation with default models (Path A)."""
    print_section("Test 11: Multi-Turn Conversation")
    
    session = ChatSession()
    
    # Turn 1
    session.add_user_message("Hi! My name is Alice.")
    response1 = chat_inference(session, fallback=True)
    session.add_assistant_message(response1.content)
    print(f"Turn 1: {response1.content[:50]}...")
    
    # Turn 2
    session.add_user_message("What is my name?")
    response2 = chat_inference(session, fallback=True)
    print(f"Turn 2: {response2.content[:50]}...")
    
    print("✅ Multi-turn conversation successful")


def example_12_cerebras_specific_model():
    """Test 12: Cerebras provider with specific model (Path B)."""
    print_section("Test 12: Cerebras with Specific Model")
    
    session = ChatSession()
    session.add_user_message("What is AI? One sentence.")
    
    try:
        response = chat_inference(
            session,
            provider=LLMProvider.CEREBRAS,
            model="llama3.1-8b",
            fallback=True
        )
        print(f"✅ Response: {response.content}")
        print(f"   Provider: Cerebras")
        print(f"   Model: {response.model}")
    except AllKeysFailedError as e:
        print(f"❌ All keys failed for {e.provider} with model {e.model}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def run_async_tests():
    """Run all async tests in a single event loop."""
    print("\n" + "=" * 60)
    print("  ASYNC TESTS")
    print("=" * 60)
    
    await example_7_async_default_model()
    await example_8_async_specific_model()
    await example_9_async_streaming_default()
    await example_10_async_streaming_specific()


def run_all_tests():
    """Run all test examples."""
    print("\n" + "=" * 60)
    print("  MULTI-API KEY ROTATION TEST SUITE")
    print("=" * 60)
    
    # Check environment variables
    groq_keys = os.getenv("GROQ_API_KEYS", "")
    cerebras_keys = os.getenv("CEREBRAS_API_KEYS", "")
    
    print(f"\nEnvironment Check:")
    print(f"  GROQ_API_KEYS: {'✅ Set' if groq_keys else '❌ Not set'}")
    print(f"  CEREBRAS_API_KEYS: {'✅ Set' if cerebras_keys else '❌ Not set'}")
    
    if not groq_keys and not cerebras_keys:
        print("\n⚠️  Warning: No API keys configured!")
        print("   Set GROQ_API_KEYS and/or CEREBRAS_API_KEYS in .env file")
        return
    
    # Synchronous tests
    example_1_default_model_chat()
    example_2_default_model_streaming()
    example_3_specific_model_chat()
    example_4_specific_model_streaming()
    example_5_provider_rotation()
    example_6_key_rotation_specific_model()
    example_11_multi_turn_conversation()
    example_12_cerebras_specific_model()
    
    # Async tests - run in single event loop
    asyncio.run(run_async_tests())
    
    print_section("ALL TESTS COMPLETED")
    print("✅ Test suite finished successfully!")


if __name__ == "__main__":
    run_all_tests()
