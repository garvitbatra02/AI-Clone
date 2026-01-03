"""
Example usage of the ChatServer LLM abstraction layer.

This file demonstrates how to use the generic LLM interface
with different providers.
"""

import os
from dotenv import load_dotenv
from ChatServer import (
    ChatSession,
    LLMConfig,
    LLMFactory,
    LLMProvider,
    GroqLLM,
)

# Load environment variables from .env file
load_dotenv()


def example_basic_usage():
    """Basic usage example with Groq."""
    # Create a chat session
    session = ChatSession()
    
    # Set up a system prompt
    session.add_system_prompt(
        "You are a helpful assistant that provides concise and accurate answers."
    )
    
    # Add user message
    session.add_user_message("What is the capital of France?")

    # Create LLM using factory (auto-detects provider from model name)
    llm = LLMFactory.create(
        provider=LLMProvider.GROQ,
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY", ""),
    )
    
    # Get response
    response = llm.chat(session)
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")
    
    # Add assistant response to session for context
    session.add_assistant_message(response.content)
    
    # Continue the conversation
    session.add_user_message("What is its population?")
    response = llm.chat(session)
    print(f"Follow-up: {response.content}")


def example_different_providers():
    """Example showing how to switch between providers."""
    session = ChatSession()
    session.add_system_prompt("You are a helpful coding assistant.")
    session.add_user_message("Write a Python function to calculate factorial.")
    
    print("\n=== Testing Different Providers ===")
    
    # Using Groq (fast inference)
    print("\n--- Using Groq ---")
    groq_llm = LLMFactory.create(
        provider=LLMProvider.GROQ,
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY", ""),
    )
    response = groq_llm.chat(session)
    print(f"Groq Response: {response.content[:200]}...")
    print(f"Tokens: {response.total_tokens}")
    
    # Using Cerebras (ultra-fast inference)
    print("\n--- Using Cerebras ---")
    cerebras_llm = LLMFactory.create(
        provider=LLMProvider.CEREBRAS,
        model="llama-3.3-70b",
        api_key=os.getenv("CEREBRAS_API_KEY", ""),
    )
    response = cerebras_llm.chat(session)
    print(f"Cerebras Response: {response.content[:200]}...")
    print(f"Tokens: {response.total_tokens}")
    
    # All LLMs use the same interface!


def example_streaming():
    """Example showing streaming responses."""
    session = ChatSession()
    session.add_system_prompt("You are a storyteller.")
    session.add_user_message("Tell me a short story about a robot.")
    
    # Using Cerebras for ultra-fast streaming
    llm = LLMFactory.create(
        provider=LLMProvider.CEREBRAS,
        model="llama3.1-8b",
        api_key=os.getenv("CEREBRAS_API_KEY", ""),
    )
    
    # Stream the response
    print("\n=== Streaming with Cerebras ===")
    print("Streaming response: ", end="")
    full_response = ""
    for chunk in llm.chat_stream(session):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    
    # Add the complete response to session
    session.add_assistant_message(full_response)


async def example_async_usage():
    """Example showing async usage."""
    session = ChatSession()
    session.add_system_prompt("You are a helpful assistant.")
    session.add_user_message("Explain quantum computing in simple terms.")
    
    # Using Cerebras for async operations
    llm = LLMFactory.create(
        provider=LLMProvider.CEREBRAS,
        model="llama-3.3-70b",
        api_key=os.getenv("CEREBRAS_API_KEY", ""),
    )
    
    # Async chat
    print("\n=== Async Usage with Cerebras ===")
    response = await llm.chat_async(session)
    print(f"Response: {response.content}")
    
    # Async streaming
    session.add_user_message("Can you give an example?")
    print("\nStreaming follow-up: ", end="")
    async for chunk in llm.chat_stream_async(session):
        print(chunk, end="", flush=True)
    print()


def example_session_management():
    """Example showing session management features."""
    # Create session with max history
    session = ChatSession(max_history_length=10)
    
    # Set system prompt
    session.add_system_prompt(
        "You are a knowledgeable assistant. Always be helpful and accurate."
    )
    
    # Add context (useful for RAG or additional information)
    session.set_context("user_name", "John")
    session.set_context("preferences", {"language": "en", "expertise": "beginner"})
    
    # Simulate a conversation
    session.add_user_message("Hello!")
    session.add_assistant_message("Hello John! How can I help you today?")
    session.add_user_message("What's the weather like?")
    session.add_assistant_message("I don't have access to real-time weather data.")
    
    # Get conversation stats
    print(f"Session ID: {session.session_id}")
    print(f"Message count: {session.get_message_count()}")
    print(f"Estimated tokens: {session.get_token_estimate()}")
    print(f"Last user message: {session.get_last_user_message().content}")
    
    # Serialize session (for persistence)
    session_json = session.to_json()
    print(f"Serialized session length: {len(session_json)} chars")
    
    # Restore session
    restored_session = ChatSession.from_json(session_json)
    print(f"Restored message count: {restored_session.get_message_count()}")


def example_direct_llm_instantiation():
    """Example showing direct LLM instantiation without factory."""
    # Create config for Groq
    config = LLMConfig(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.8,
        max_tokens=2048,
        top_p=0.9,
    )
    
    # Direct instantiation
    llm = GroqLLM(config)
    
    # Create session and use
    session = ChatSession()
    session.add_system_prompt("You are a creative writing assistant.")
    session.add_user_message("Write a haiku about programming.")
    
    # response = llm.chat(session)
    # print(response.content)


def example_session_persistence_workflow():
    """
    Example showing a complete workflow with session persistence.
    This is useful for chatbot applications where sessions need to be saved.
    """
    # --- First interaction ---
    session = ChatSession()
    session.add_system_prompt("You are a helpful travel advisor.")
    session.add_user_message("I want to plan a trip to Japan.")
    
    # Save session state (e.g., to database)
    session_data = session.to_dict()
    # save_to_database(session.session_id, session_data)
    
    # --- Later, when user returns ---
    # session_data = load_from_database(session_id)
    # restored_session = ChatSession.from_dict(session_data)
    
    # Continue conversation
    # restored_session.add_user_message("What's the best time to visit Tokyo?")
    # response = llm.chat(restored_session)


if __name__ == "__main__":
    # Run examples (uncomment as needed)
    print("\n" + "="*60)
    print("ChatServer LLM Examples - Groq & Cerebras AI")
    print("="*60)
    
    # example_basic_usage()
    # example_different_providers()
    # example_streaming()
    # example_session_management()
    example_direct_llm_instantiation()
    
    # For async examples:
    # import asyncio
    # asyncio.run(example_async_usage())
