"""
Model Registry for LLM Providers.

This module contains the complete mapping of all available models for each provider.
Each model is explicitly listed with its provider for deterministic model detection.
"""

from typing import Dict
from .base import LLMProvider


# Complete registry of all available models and their providers
MODEL_REGISTRY: Dict[str, LLMProvider] = {
    # ===== GOOGLE GEMINI MODELS =====
    # Gemini 3 Series (Latest Generation)
    "gemini-3-pro": LLMProvider.GEMINI,
    "gemini-3-flash": LLMProvider.GEMINI,
    
    # Gemini 2.5 Series
    "gemini-2.5-flash": LLMProvider.GEMINI,
    "gemini-2.5-flash-lite": LLMProvider.GEMINI,
    "gemini-2.5-pro": LLMProvider.GEMINI,
    
    # Gemini 2.0 Series
    "gemini-2.0-flash": LLMProvider.GEMINI,
    "gemini-2.0-flash-lite": LLMProvider.GEMINI,
    
    # Gemini 1.5 Series (Legacy)
    "gemini-1.5-pro": LLMProvider.GEMINI,
    "gemini-1.5-pro-latest": LLMProvider.GEMINI,
    "gemini-1.5-flash": LLMProvider.GEMINI,
    "gemini-1.5-flash-latest": LLMProvider.GEMINI,
    "gemini-1.5-flash-8b": LLMProvider.GEMINI,
    
    # Gemini 1.0 Series (Legacy)
    "gemini-pro": LLMProvider.GEMINI,
    "gemini-pro-vision": LLMProvider.GEMINI,
    
    # Gemini Latest Aliases
    "gemini-flash-latest": LLMProvider.GEMINI,
    "gemini-pro-latest": LLMProvider.GEMINI,
    
    # ===== GROQ MODELS =====
    # Production Models
    "llama-3.1-8b-instant": LLMProvider.GROQ,
    "llama-3.3-70b-versatile": LLMProvider.GROQ,
    "meta-llama/llama-guard-4-12b": LLMProvider.GROQ,
    "openai/gpt-oss-120b": LLMProvider.GROQ,
    "openai/gpt-oss-20b": LLMProvider.GROQ,
    "whisper-large-v3": LLMProvider.GROQ,
    "whisper-large-v3-turbo": LLMProvider.GROQ,
    
    # Production Systems
    "groq/compound": LLMProvider.GROQ,
    "groq/compound-mini": LLMProvider.GROQ,
    
    # Preview Models
    "canopylabs/orpheus-arabic-saudi": LLMProvider.GROQ,
    "canopylabs/orpheus-v1-english": LLMProvider.GROQ,
    "meta-llama/llama-4-maverick-17b-128e-instruct": LLMProvider.GROQ,
    "meta-llama/llama-4-scout-17b-16e-instruct": LLMProvider.GROQ,
    "meta-llama/llama-prompt-guard-2-22m": LLMProvider.GROQ,
    "meta-llama/llama-prompt-guard-2-86m": LLMProvider.GROQ,
    "moonshotai/kimi-k2-instruct-0905": LLMProvider.GROQ,
    "openai/gpt-oss-safeguard-20b": LLMProvider.GROQ,
    "qwen/qwen3-32b": LLMProvider.GROQ,
    
    # Legacy Groq Models (Common naming patterns)
    "llama-3-8b": LLMProvider.GROQ,
    "llama-3-70b": LLMProvider.GROQ,
    "llama2-70b-4096": LLMProvider.GROQ,
    "mixtral-8x7b-32768": LLMProvider.GROQ,
    "gemma-7b-it": LLMProvider.GROQ,
    "gemma2-9b-it": LLMProvider.GROQ,
    
    # ===== CEREBRAS MODELS =====
    # Production Models
    "llama3.1-8b": LLMProvider.CEREBRAS,
    "gpt-oss-120b": LLMProvider.CEREBRAS,
    
    # ===== COHERE MODELS =====
    # Command A Series (Latest Generation)
    "command-a-03-2025": LLMProvider.COHERE,
    
    # Command R Series
    "command-r-08-2024": LLMProvider.COHERE,
    "command-r-plus-08-2024": LLMProvider.COHERE,
    
    # Command R7B (Lightweight)
    "command-r7b-12-2024": LLMProvider.COHERE,
    
    # Legacy Command Models
    "command": LLMProvider.COHERE,
    "command-light": LLMProvider.COHERE,
    "command-nightly": LLMProvider.COHERE,
    
    # ===== NVIDIA NIM MODELS =====
    # Nemotron 3 Series (Flagship - Hybrid Mamba-Transformer MoE)
    "nvidia/nemotron-3-ultra-550b-a55b": LLMProvider.NVIDIA,
    "nvidia/nemotron-3-super-120b-a12b": LLMProvider.NVIDIA,
    "nvidia/nemotron-3-nano-30b-a3b": LLMProvider.NVIDIA,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": LLMProvider.NVIDIA,
    
    # Nemotron Legacy
    "nvidia/nvidia-nemotron-nano-9b-v2": LLMProvider.NVIDIA,
    "nvidia/nemotron-mini-4b-instruct": LLMProvider.NVIDIA,
    
    # Llama-Nemotron Series (NVIDIA fine-tuned)
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": LLMProvider.NVIDIA,
    "nvidia/llama-3.3-nemotron-super-49b-v1": LLMProvider.NVIDIA,
    "nvidia/llama-3.1-nemotron-nano-8b-v1": LLMProvider.NVIDIA,
    
    # Meta Llama on NVIDIA NIM
    "meta/llama-3.3-70b-instruct": LLMProvider.NVIDIA,
    "meta/llama-3.2-3b-instruct": LLMProvider.NVIDIA,
    "meta/llama-3.2-1b-instruct": LLMProvider.NVIDIA,
    "meta/llama-4-maverick-17b-128e-instruct": LLMProvider.NVIDIA,
    
    # Mistral on NVIDIA NIM
    "mistralai/mistral-medium-3.5-128b": LLMProvider.NVIDIA,
    "mistralai/mistral-small-4-119b-2603": LLMProvider.NVIDIA,
    "mistralai/mistral-large-3-675b-instruct-2512": LLMProvider.NVIDIA,
    "mistralai/ministral-14b-instruct-2512": LLMProvider.NVIDIA,
    "mistralai/mistral-nemotron": LLMProvider.NVIDIA,
    
    # DeepSeek on NVIDIA NIM
    "deepseek-ai/deepseek-v4-flash": LLMProvider.NVIDIA,
    "deepseek-ai/deepseek-v4-pro": LLMProvider.NVIDIA,
    
    # Qwen on NVIDIA NIM
    "qwen/qwen3.5-122b-a10b": LLMProvider.NVIDIA,
    "qwen/qwen3.5-397b-a17b": LLMProvider.NVIDIA,
    "qwen/qwen3-next-80b-a3b-instruct": LLMProvider.NVIDIA,
    
    # Google on NVIDIA NIM
    "google/gemma-4-31b-it": LLMProvider.NVIDIA,
    "google/gemma-3n-e4b-it": LLMProvider.NVIDIA,
    "google/gemma-3n-e2b-it": LLMProvider.NVIDIA,
    "google/diffusiongemma-26b-a4b-it": LLMProvider.NVIDIA,
    
    # Other providers on NVIDIA NIM
    "moonshotai/kimi-k2.6": LLMProvider.NVIDIA,
    "z-ai/glm-5.1": LLMProvider.NVIDIA,
    "stepfun-ai/step-3.7-flash": LLMProvider.NVIDIA,
    "stepfun-ai/step-3.5-flash": LLMProvider.NVIDIA,
    "minimaxai/minimax-m3": LLMProvider.NVIDIA,
    "minimaxai/minimax-m2.7": LLMProvider.NVIDIA,
    "bytedance/seed-oss-36b-instruct": LLMProvider.NVIDIA,
    "microsoft/phi-4-mini-instruct": LLMProvider.NVIDIA,
    "openai/gpt-oss-20b": LLMProvider.NVIDIA,
    "openai/gpt-oss-120b": LLMProvider.NVIDIA,
    "abacusai/dracarys-llama-3.1-70b-instruct": LLMProvider.NVIDIA,
    
    # ===== MISTRAL AI MODELS =====
    # Mistral Large (Flagship)
    "mistral-large-latest": LLMProvider.MISTRAL,
    "mistral-large-2512": LLMProvider.MISTRAL,
    
    # Mistral Medium (Balanced)
    "mistral-medium-latest": LLMProvider.MISTRAL,
    "mistral-medium-2604": LLMProvider.MISTRAL,
    "mistral-medium-2508": LLMProvider.MISTRAL,
    "mistral-medium-2505": LLMProvider.MISTRAL,
    "mistral-medium-3.5": LLMProvider.MISTRAL,
    "mistral-medium-3": LLMProvider.MISTRAL,
    "mistral-medium": LLMProvider.MISTRAL,
    
    # Mistral Small (Efficient)
    "mistral-small-latest": LLMProvider.MISTRAL,
    "mistral-small-2603": LLMProvider.MISTRAL,
    "mistral-small-2506": LLMProvider.MISTRAL,
    
    # Ministral (Edge/Lightweight)
    "ministral-8b-latest": LLMProvider.MISTRAL,
    "ministral-8b-2512": LLMProvider.MISTRAL,
    "ministral-3b-latest": LLMProvider.MISTRAL,
    "ministral-3b-2512": LLMProvider.MISTRAL,
    "ministral-14b-latest": LLMProvider.MISTRAL,
    "ministral-14b-2512": LLMProvider.MISTRAL,
    
    # Codestral (Code Generation)
    "codestral-latest": LLMProvider.MISTRAL,
    "codestral-2508": LLMProvider.MISTRAL,
    
    # Magistral (Reasoning)
    "magistral-medium-latest": LLMProvider.MISTRAL,
    "magistral-medium-2509": LLMProvider.MISTRAL,
    "magistral-small-latest": LLMProvider.MISTRAL,
    "magistral-small-2509": LLMProvider.MISTRAL,
    
    # Devstral (Development)
    "devstral-latest": LLMProvider.MISTRAL,
    "devstral-2512": LLMProvider.MISTRAL,
    "devstral-medium-latest": LLMProvider.MISTRAL,
    
    # Open-weight Models
    "open-mistral-nemo": LLMProvider.MISTRAL,
    "open-mistral-nemo-2407": LLMProvider.MISTRAL,
    
    # Tiny (Legacy)
    "mistral-tiny-latest": LLMProvider.MISTRAL,
    "mistral-tiny-2407": LLMProvider.MISTRAL,
}


def get_provider_for_model(model: str) -> LLMProvider | None:
    """
    Get the provider for a given model name.
    
    This function performs a deterministic lookup in the model registry.
    Unlike prefix matching, this ensures exact model name matching.
    
    Args:
        model: The exact model name
        
    Returns:
        The LLMProvider enum value if found, None otherwise
        
    Example:
        >>> get_provider_for_model("gemini-2.5-flash")
        <LLMProvider.GEMINI: 'gemini'>
        >>> get_provider_for_model("llama3.1-8b")
        <LLMProvider.CEREBRAS: 'cerebras'>
        >>> get_provider_for_model("unknown-model")
        None
    """
    return MODEL_REGISTRY.get(model)


def get_all_models_for_provider(provider: LLMProvider) -> list[str]:
    """
    Get all available models for a specific provider.
    
    Args:
        provider: The LLM provider
        
    Returns:
        List of model names for the given provider
        
    Example:
        >>> get_all_models_for_provider(LLMProvider.GEMINI)
        ['gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-flash', ...]
    """
    return [model for model, prov in MODEL_REGISTRY.items() if prov == provider]


def is_model_supported(model: str) -> bool:
    """
    Check if a model is supported by any provider.
    
    Args:
        model: The model name to check
        
    Returns:
        True if the model is in the registry, False otherwise
        
    Example:
        >>> is_model_supported("gemini-2.5-flash")
        True
        >>> is_model_supported("gpt-4")
        False
    """
    return model in MODEL_REGISTRY


def get_all_supported_models() -> list[str]:
    """
    Get a list of all supported models across all providers.
    
    Returns:
        List of all model names in the registry
    """
    return list(MODEL_REGISTRY.keys())


def get_provider_model_count() -> Dict[LLMProvider, int]:
    """
    Get the count of models for each provider.
    
    Returns:
        Dictionary mapping providers to their model counts
        
    Example:
        >>> get_provider_model_count()
        {<LLMProvider.GEMINI: 'gemini'>: 18, <LLMProvider.GROQ: 'groq'>: 23, ...}
    """
    counts = {}
    for provider in LLMProvider:
        counts[provider] = len(get_all_models_for_provider(provider))
    return counts
