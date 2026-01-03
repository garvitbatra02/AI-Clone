"""
Example usage of ChatServer's ChatService with provider rotation and fallback.

This demonstrates:
- Provider rotation between Groq and Cerebras
- Automatic fallback on provider failure
- Error handling with AllProvidersFailedError
- Streaming responses
- Both sync and async usage
"""

import os
import asyncio
from dotenv import load_dotenv
from ChatServer import (
    ChatSession,
    ChatService,
    ProviderConfig,
    LLMProvider,
    get_chat_service,
    chat_inference,
    chat_inference_stream,
    chat_inference_async,
    chat_inference_stream_async,
    AllProvidersFailedError,
)

# Load environment variables
load_dotenv()


def example_basic_rotation():
    """Example showing automatic provider rotation."""
    print("\n" + "="*60)
    print("Example 1: Basic Provider Rotation")
    print("="*60)
    
    # Create sessions for multiple requests
    sessions = []
    for i in range(4):
        session = ChatSession()
        session.add_system_prompt("You are a helpful assistant. Be brief.")
        session.add_user_message(f"What is {i + 1} + {i + 1}?")
        sessions.append(session)
    
    # Make multiple requests - they will rotate between providers
    service = get_chat_service()
    
    for i, session in enumerate(sessions):
        try:
            print(f"\nRequest {i + 1}:")
            response = service.chat(session, fallback=False)
            print(f"  Response: {response.content[:100]}")
            print(f"  Tokens: {response.total_tokens}")
        except Exception as e:
            print(f"  Error: {e}")


def example_with_fallback():
    """Example showing automatic fallback on provider failure."""
    print("\n" + "="*60)
    print("Example 2: Automatic Fallback (with valid providers)")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a helpful math tutor.")
    session.add_user_message("Explain the Pythagorean theorem in one sentence.")
    
    try:
        # With fallback enabled (default) - will try all providers before failing
        print("\nUsing chat_inference with fallback=True (default)...")
        response = chat_inference(session, fallback=True)
        print(f"Success! Response: {response.content[:150]}...")
        print(f"Tokens used: {response.total_tokens}")
    except AllProvidersFailedError as e:
        print(f"All providers failed: {e.errors}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_without_fallback():
    """Example showing behavior without fallback."""
    print("\n" + "="*60)
    print("Example 3: Without Fallback (fails immediately)")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a coding assistant.")
    session.add_user_message("Write a hello world in Python.")
    
    try:
        # Without fallback - fails immediately if provider has issues
        print("\nUsing chat_inference with fallback=False...")
        response = chat_inference(session, fallback=False)
        print(f"Success! Response: {response.content[:100]}...")
    except Exception as e:
        print(f"Failed immediately: {type(e).__name__}: {str(e)[:100]}")


def example_streaming_with_fallback():
    """Example showing streaming with fallback."""
    print("\n" + "="*60)
    print("Example 4: Streaming with Fallback")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a creative writer.")
    session.add_user_message("Write a two-sentence story about a robot.")
    
    try:
        print("\nStreaming response (with fallback):")
        print("Response: ", end="", flush=True)
        
        full_response = ""
        for chunk in chat_inference_stream(session, fallback=True):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()
        print(f"\nTotal characters: {len(full_response)}")
    except AllProvidersFailedError as e:
        print(f"\nAll providers failed for streaming: {e.errors}")
    except Exception as e:
        print(f"\nStreaming error: {e}")


async def example_async_usage():
    """Example showing async usage with fallback."""
    print("\n" + "="*60)
    print("Example 5: Async Usage with Fallback")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a helpful assistant.")
    session.add_user_message("What is the speed of light? Answer in one sentence.")
    
    try:
        print("\nMaking async request...")
        response = await chat_inference_async(session, fallback=True)
        print(f"Success! Response: {response.content}")
        print(f"Tokens: {response.total_tokens}")
    except AllProvidersFailedError as e:
        print(f"All providers failed: {e.errors}")
    except Exception as e:
        print(f"Async error: {e}")


async def example_async_streaming():
    """Example showing async streaming with fallback."""
    print("\n" + "="*60)
    print("Example 6: Async Streaming with Fallback")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a poet.")
    session.add_user_message("Write a short haiku about coding.")
    
    try:
        print("\nStreaming async response:")
        print("Response: ", end="", flush=True)
        
        full_response = ""
        async for chunk in chat_inference_stream_async(session, fallback=True):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()
        print(f"\nTotal characters: {len(full_response)}")
    except AllProvidersFailedError as e:
        print(f"\nAll providers failed: {e.errors}")
    except Exception as e:
        print(f"\nAsync streaming error: {e}")


def example_service_management():
    """Example showing service management features."""
    print("\n" + "="*60)
    print("Example 7: Service Management")
    print("="*60)
    
    service = get_chat_service()
    
    print(f"\nActive providers: {service.get_active_providers()}")
    print(f"Provider count: {service.get_provider_count()}")
    print(f"Current provider: {service.get_current_provider().provider.value}")
    
    # Make a request to see rotation
    session = ChatSession()
    session.add_system_prompt("You are a helpful assistant.")
    session.add_user_message("Say 'hello' in one word.")
    
    print("\nMaking request...")
    try:
        response = service.chat(session, fallback=False)
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Current provider after request: {service.get_current_provider().provider.value}")


def example_specific_provider():
    """Example showing how to use a specific provider."""
    print("\n" + "="*60)
    print("Example 8: Using Specific Provider")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a helpful assistant.")
    session.add_user_message("What is 2+2?")
    
    # Use Groq specifically
    print("\nUsing Groq specifically:")
    try:
        response = chat_inference(session, provider=LLMProvider.GROQ, fallback=False)
        print(f"Groq response: {response.content[:100]}")
    except Exception as e:
        print(f"Groq error: {e}")
    
    # Use Cerebras specifically
    print("\nUsing Cerebras specifically:")
    try:
        response = chat_inference(session, provider=LLMProvider.CEREBRAS, fallback=False)
        print(f"Cerebras response: {response.content[:100]}")
    except Exception as e:
        print(f"Cerebras error: {e}")


def example_error_handling():
    """Example showing comprehensive error handling."""
    print("\n" + "="*60)
    print("Example 9: Comprehensive Error Handling")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a helpful assistant.")
    session.add_user_message("Tell me a fun fact.")
    
    try:
        # Try with fallback enabled
        response = chat_inference(session, fallback=True)
        print(f"Success! Response: {response.content[:100]}...")
        print(f"Tokens: {response.total_tokens}")
    except AllProvidersFailedError as e:
        # This error contains details about all provider failures
        print("\n❌ All providers failed to respond:")
        for provider, error in e.errors.items():
            print(f"  - {provider}: {error}")
        print("\nYou might want to:")
        print("  1. Check your API keys")
        print("  2. Verify your internet connection")
        print("  3. Check provider status pages")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")


def example_single_session_multi_turn():
    """
    Example showing rotation with SINGLE SESSION and multiple turns.
    
    This demonstrates that rotation happens at ChatService level, not session level.
    Each API call rotates to next provider, even within the same conversation.
    """
    print("\n" + "="*60)
    print("Example 10: Single Session, Multiple Turns with Rotation")
    print("="*60)
    print("\nDemonstrates: ONE user/session, MULTIPLE API calls, AUTOMATIC ROTATION")
    print("Each message alternates between Groq → Cerebras → Groq → Cerebras...")
    
    # ← SINGLE SESSION - represents one user
    session = ChatSession()
    session.add_system_prompt("You are a helpful math tutor. Keep answers brief.")
    
    # Multiple questions from the same user
    questions = [
        "What is 5 + 3?",
        "What is 12 - 7?",
        "What is 8 * 6?",
        "What is 20 / 4?",
    ]
    
    service = get_chat_service()
    
    for i, question in enumerate(questions, 1):
        session.add_user_message(question)
        
        try:
            # Get current provider before the call
            provider_before = service.get_current_provider().provider.value
            
            print(f"\n{i}. Q: {question}")
            
            # Make the API call
            response = chat_inference(session, fallback=True)
            
            # Get current provider after the call (it has rotated)
            provider_after = service.get_current_provider().provider.value
            
            print(f"   Provider used: {provider_before}")
            print(f"   A: {response.content}")
            print(f"   Next provider: {provider_after}")
            
            # Add assistant response to maintain conversation context
            session.add_assistant_message(response.content)
        except AllProvidersFailedError as e:
            print(f"   Error: All providers failed - {e}")
            break
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    print(f"\n--- Session Statistics ---")
    print(f"Total messages: {session.get_message_count()}")
    print(f"Estimated tokens: {session.get_token_estimate()}")
    print(f"\nKey Point: Same session (user), but each API call used different provider!")


def example_conversation_with_rotation():
    """Example showing a multi-turn conversation with rotation."""
    print("\n" + "="*60)
    print("Example 11: Multi-turn Conversation with Rotation")
    print("="*60)
    
    session = ChatSession()
    session.add_system_prompt("You are a knowledgeable assistant. Be concise.")
    
    questions = [
        "What is the capital of France?",
        "What is its population?",
        "Name one famous landmark there.",
    ]
    
    for i, question in enumerate(questions, 1):
        session.add_user_message(question)
        
        try:
            print(f"\n{i}. Q: {question}")
            response = chat_inference(session, fallback=True)
            print(f"   A: {response.content}")
            
            # Add assistant response to maintain conversation context
            session.add_assistant_message(response.content)
        except AllProvidersFailedError as e:
            print(f"   Error: All providers failed - {e}")
            break
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    print(f"\nConversation stats:")
    print(f"  - Total messages: {session.get_message_count()}")
    print(f"  - Estimated tokens: {session.get_token_estimate()}")


async def run_async_examples():
    """Run all async examples."""
    await example_async_usage()
    await example_async_streaming()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ChatServer Service Examples - Provider Rotation & Fallback")
    print("="*60)
    
    # Run sync examples
    # example_basic_rotation()
    # example_single_session_multi_turn()  # ← NEW: Single session, multiple turns
    # example_with_fallback()
    # example_without_fallback()
    # example_streaming_with_fallback()
    # example_service_management()
    # example_specific_provider()
    # example_error_handling()
    # example_conversation_with_rotation()
    
    # Run async examples
    print("\n" + "="*60)
    print("Running Async Examples...")
    print("="*60)
    asyncio.run(run_async_examples())
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
 