"""API Key Manager for rotating API keys per provider."""

import os
import threading
from typing import Dict, List
from .base import LLMProvider


class APIKeyManager:
    """Manages API key rotation for LLM providers.
    
    Provides thread-safe round-robin rotation of API keys for each provider.
    API keys are loaded from environment variables as comma-separated strings.
    """
    
    def __init__(self):
        """Initialize the API key manager with rotation state."""
        self._current_indices: Dict[LLMProvider, int] = {}
        self._locks: Dict[LLMProvider, threading.Lock] = {}
        self._api_keys_cache: Dict[LLMProvider, List[str]] = {}
        
    def get_api_keys(self, provider: LLMProvider, env_var_name: str) -> List[str]:
        """Get list of API keys for a provider from environment variable.
        
        Args:
            provider: The LLM provider
            env_var_name: Name of environment variable containing comma-separated keys
            
        Returns:
            List of API keys (empty list if env var not set or empty)
            
        Example:
            GROQ_API_KEYS="key1,key2,key3" -> ["key1", "key2", "key3"]
        """
        # Check cache first
        if provider in self._api_keys_cache:
            return self._api_keys_cache[provider]
        
        # Load from environment
        keys_str = os.getenv(env_var_name, "")
        api_keys = [key.strip() for key in keys_str.split(",") if key.strip()]
        
        # Cache the result
        self._api_keys_cache[provider] = api_keys
        
        # Initialize rotation state for this provider
        if provider not in self._current_indices:
            self._current_indices[provider] = 0
        if provider not in self._locks:
            self._locks[provider] = threading.Lock()
            
        return api_keys
    
    def get_next_key_index(self, provider: LLMProvider) -> int:
        """Get the next API key index for a provider using round-robin rotation.
        
        Args:
            provider: The LLM provider
            
        Returns:
            Index of the next API key to use
            
        Thread-safe: Uses a lock to ensure atomic rotation.
        """
        # Ensure provider is initialized
        if provider not in self._locks:
            self._locks[provider] = threading.Lock()
        if provider not in self._current_indices:
            self._current_indices[provider] = 0
            
        with self._locks[provider]:
            current_index = self._current_indices[provider]
            # Get number of keys for this provider from cache
            num_keys = len(self._api_keys_cache.get(provider, []))
            if num_keys > 0:
                self._current_indices[provider] = (current_index + 1) % num_keys
            return current_index
    
    def get_key_by_index(self, provider: LLMProvider, index: int, env_var_name: str) -> str:
        """Get a specific API key by index for a provider.
        
        Args:
            provider: The LLM provider
            index: Index of the key to retrieve
            env_var_name: Name of environment variable containing keys
            
        Returns:
            The API key at the specified index
            
        Raises:
            IndexError: If index is out of range
            ValueError: If no API keys are configured for the provider
        """
        api_keys = self.get_api_keys(provider, env_var_name)
        if not api_keys:
            raise ValueError(f"No API keys configured for {provider.value} (env var: {env_var_name})")
        if index < 0 or index >= len(api_keys):
            raise IndexError(f"Key index {index} out of range for {provider.value} (available: 0-{len(api_keys)-1})")
        return api_keys[index]
    
    def reset_rotation(self, provider: LLMProvider):
        """Reset rotation index for a provider back to 0.
        
        Args:
            provider: The LLM provider to reset
        """
        if provider in self._locks:
            with self._locks[provider]:
                self._current_indices[provider] = 0
    
    def clear_cache(self):
        """Clear the API keys cache. Useful for testing or reloading configuration."""
        self._api_keys_cache.clear()


# Global singleton instance
_api_key_manager_instance = None
_api_key_manager_lock = threading.Lock()


def get_api_key_manager() -> APIKeyManager:
    """Get the global APIKeyManager singleton instance.
    
    Returns:
        The global APIKeyManager instance
    """
    global _api_key_manager_instance
    if _api_key_manager_instance is None:
        with _api_key_manager_lock:
            if _api_key_manager_instance is None:
                _api_key_manager_instance = APIKeyManager()
    return _api_key_manager_instance
