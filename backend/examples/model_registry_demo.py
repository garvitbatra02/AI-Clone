"""
Example script demonstrating the Model Registry functionality.

This script shows how to:
1. Check supported models
2. Look up providers for specific models
3. Get all models for a provider
4. Use the registry with LLMFactory

Run this script to see the model registry in action.
"""

from ChatService.Chat import (
    LLMFactory,
    LLMProvider,
    get_provider_for_model,
    get_all_models_for_provider,
    is_model_supported,
    get_all_supported_models,
    get_provider_model_count,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_model_lookup():
    """Demonstrate looking up providers for specific models."""
    print_section("1. Model Lookup")
    
    test_models = [
        "gemini-2.5-flash",
        "llama-3.3-70b",  # Cerebras version
        "llama-3.3-70b-versatile",  # Groq version
        "gpt-oss-120b",  # Cerebras
        "openai/gpt-oss-120b",  # Groq
        "unknown-model",  # Not in registry
    ]
    
    for model in test_models:
        provider = get_provider_for_model(model)
        if provider:
            print(f"✓ {model:<35} → {provider.value}")
        else:
            print(f"✗ {model:<35} → Not supported")


def demo_provider_models():
    """Show all models for each provider."""
    print_section("2. Models by Provider")
    
    for provider in LLMProvider:
        models = get_all_models_for_provider(provider)
        print(f"\n{provider.value.upper()} ({len(models)} models):")
        print("-" * 40)
        
        # Show first 5 models as examples
        for model in models[:5]:
            print(f"  • {model}")
        
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")


def demo_model_statistics():
    """Display statistics about supported models."""
    print_section("3. Model Statistics")
    
    # Count by provider
    counts = get_provider_model_count()
    total = sum(counts.values())
    
    print("Models per provider:")
    for provider, count in counts.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {provider.value:<12} {count:>3} models  {bar} {percentage:.1f}%")
    
    print(f"\n  {'TOTAL':<12} {total:>3} models")


def demo_model_validation():
    """Demonstrate model validation."""
    print_section("4. Model Validation")
    
    test_cases = [
        ("gemini-2.5-flash", True),
        ("gpt-4", False),
        ("llama-3.3-70b", True),
        ("claude-3", False),
    ]
    
    print("Testing model support:")
    for model, expected in test_cases:
        is_supported = is_model_supported(model)
        status = "✓" if is_supported == expected else "✗"
        result = "SUPPORTED" if is_supported else "NOT SUPPORTED"
        print(f"  {status} {model:<25} → {result}")


def demo_factory_usage():
    """Show how to use the factory with model registry."""
    print_section("5. Factory Usage with Auto-Detection")
    
    # Note: These examples show the API usage but won't actually create
    # LLM instances without valid API keys
    
    examples = [
        ("gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("llama-3.3-70b", "CEREBRAS_API_KEY"),
        ("qwen/qwen3-32b", "GROQ_API_KEY"),
    ]
    
    print("Example factory usage (requires API keys):\n")
    
    for model, key_name in examples:
        provider = get_provider_for_model(model)
        if provider:
            print(f"Model: {model}")
            print(f"Provider: {provider.value}")
            print(f"Code:")
            print(f'  llm = LLMFactory.from_model(')
            print(f'      model="{model}",')
            print(f'      api_key=os.getenv("{key_name}")')
            print(f'  )')
            print()


def demo_ambiguous_models():
    """Show how the registry handles models with similar names."""
    print_section("6. Handling Ambiguous Model Names")
    
    print("The registry handles models with similar names across providers:\n")
    
    # Llama models
    print("Llama 3.3 70B variants:")
    print(f"  • llama-3.3-70b (Cerebras)           → {get_provider_for_model('llama-3.3-70b')}")
    print(f"  • llama-3.3-70b-versatile (Groq)     → {get_provider_for_model('llama-3.3-70b-versatile')}")
    
    print("\nOpenAI GPT OSS models:")
    print(f"  • gpt-oss-120b (Cerebras)            → {get_provider_for_model('gpt-oss-120b')}")
    print(f"  • openai/gpt-oss-120b (Groq)         → {get_provider_for_model('openai/gpt-oss-120b')}")
    
    print("\nLlama 3.1 8B variants:")
    print(f"  • llama3.1-8b (Cerebras)             → {get_provider_for_model('llama3.1-8b')}")
    print(f"  • llama-3.1-8b-instant (Groq)        → {get_provider_for_model('llama-3.1-8b-instant')}")


def demo_search_models():
    """Demonstrate searching for models by name pattern."""
    print_section("7. Searching for Models")
    
    all_models = get_all_supported_models()
    
    search_terms = ["gemini-2", "llama-3.3", "qwen"]
    
    for term in search_terms:
        matches = [m for m in all_models if term in m]
        print(f"\nModels containing '{term}' ({len(matches)} found):")
        for model in matches[:5]:  # Show first 5
            provider = get_provider_for_model(model)
            print(f"  • {model:<40} ({provider.value})")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  MODEL REGISTRY DEMONSTRATION")
    print("=" * 60)
    
    demo_model_lookup()
    demo_provider_models()
    demo_model_statistics()
    demo_model_validation()
    demo_factory_usage()
    demo_ambiguous_models()
    demo_search_models()
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
