# Multi-API Key Rotation System

Complete implementation of multi-API key rotation with selective caching and model flexibility.

## Architecture Overview

### Two Execution Paths

#### Path A: Default Models (Cached with Provider Rotation)
- **Trigger**: No `model` parameter in request
- **Behavior**:
  - Uses default models per provider (configured in `DEFAULT_PROVIDERS`)
  - Groq: `llama-3.3-70b-versatile`
  - Cerebras: `llama-3.3-70b`
  - Rotates between providers: Groq → Cerebras → Groq...
  - Caches LLM instances per `(provider, key_index)`
  - Fallback tries all providers before failing
- **Use Case**: General purpose chat with automatic load balancing

#### Path B: Specific Models (On-Demand with Key-Level Fallback)
- **Trigger**: `model` parameter specified in request
- **Behavior**:
  - Uses exact model requested
  - Rotates through API keys for that provider
  - Creates LLM instances on-demand (not cached)
  - Fallback tries all API keys for that model
  - Does NOT switch providers or models
- **Use Case**: When you need a specific model's capabilities

## Environment Configuration

### .env File Format

```bash
# Comma-separated API keys for each provider
GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3"
CEREBRAS_API_KEYS="csk_key1,csk_key2,csk_key3"
```

### Single Key Configuration (Basic)
```bash
GROQ_API_KEYS="gsk_abc123"
CEREBRAS_API_KEYS="csk_xyz789"
```

### Multiple Keys Configuration (Recommended)
```bash
GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3,gsk_key4"
CEREBRAS_API_KEYS="csk_key1,csk_key2"
```

## API Usage

### REST API Examples

#### 1. Default Model (Path A)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "fallback": true
  }'
```

#### 2. Specific Model (Path B)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "model": "llama-3.1-8b-instant",
    "provider": "groq",
    "fallback": true
  }'
```

#### 3. Streaming with Default Model
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count to 10"}
    ],
    "fallback": true
  }'
```

#### 4. Streaming with Specific Model
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "model": "llama-3.3-70b-versatile",
    "provider": "groq",
    "fallback": true
  }'
```

### Python SDK Usage

#### Default Model Chat
```python
from ChatServer import ChatSession, chat_inference

session = ChatSession()
session.add_user_message("Hello!")

# Uses default models with provider rotation
response = chat_inference(session, fallback=True)
print(response.content)
```

#### Specific Model Chat
```python
from ChatServer import ChatSession, LLMProvider, chat_inference

session = ChatSession()
session.add_user_message("Explain quantum computing")

# Uses specific model with key-level fallback
response = chat_inference(
    session,
    provider=LLMProvider.GROQ,
    model="llama-3.1-8b-instant",
    fallback=True
)
print(response.content)
```

#### Streaming with Default Model
```python
from ChatServer import ChatSession, chat_inference_stream

session = ChatSession()
session.add_user_message("Write a poem")

for chunk in chat_inference_stream(session, fallback=True):
    print(chunk, end="", flush=True)
```

#### Streaming with Specific Model
```python
from ChatServer import ChatSession, LLMProvider, chat_inference_stream

session = ChatSession()
session.add_user_message("Tell me a story")

for chunk in chat_inference_stream(
    session,
    provider=LLMProvider.GROQ,
    model="llama-3.3-70b-versatile",
    fallback=True
):
    print(chunk, end="", flush=True)
```

#### Async Operations
```python
import asyncio
from ChatServer import ChatSession, chat_inference_async

async def chat():
    session = ChatSession()
    session.add_user_message("Hello!")
    
    response = await chat_inference_async(session, fallback=True)
    print(response.content)

asyncio.run(chat())
```

## Error Handling

### AllProvidersFailedError (Path A)
```python
from ChatServer import chat_inference, AllProvidersFailedError

try:
    response = chat_inference(session, fallback=True)
except AllProvidersFailedError as e:
    print(f"All providers failed: {e.errors}")
    # e.errors = {"groq": "Rate limit", "cerebras": "API error"}
```

### AllKeysFailedError (Path B)
```python
from ChatServer import chat_inference
from ChatServer.services.chat_service import AllKeysFailedError

try:
    response = chat_inference(
        session,
        model="llama-3.1-8b-instant",
        fallback=True
    )
except AllKeysFailedError as e:
    print(f"All keys failed for {e.provider} with {e.model}")
    print(f"Errors: {e.errors}")
    # e.errors = {0: "Rate limit", 1: "API error", 2: "Timeout"}
```

## Key Rotation Behavior

### Default Model (Path A)
1. Request comes in without model parameter
2. Get next provider in rotation (round-robin)
3. Get next API key for that provider (round-robin)
4. Check cache for `(provider, key_index)` instance
5. If cached, reuse; if not, create and cache
6. If fallback enabled and request fails, try next provider

### Specific Model (Path B)
1. Request comes in with model parameter
2. Create temporary LLM instance with first API key
3. If fallback enabled and request fails:
   - Try next API key for same provider
   - Repeat until all keys exhausted
4. Never switches to different provider or model

### Streaming Key-Level Fallback
1. Create LLM instance with current API key
2. Start streaming and get first chunk
3. If first chunk succeeds, commit to stream
4. If first chunk fails, try next API key
5. Once streaming starts, cannot switch keys

## Caching Strategy

### What Gets Cached
- **Default models only**: LLM instances for default models
- **Cache key**: `(provider, key_index)` tuple
- **Estimated size**: ~6 instances (2 providers × 3 keys)

### What Doesn't Get Cached
- **Specific models**: Created on-demand, not cached
- **Reason**: Avoid memory bloat (6 models × 3 keys × 2 providers = 36 instances)

## Testing

### Run All Tests
```bash
cd backend
python tests/test_multi_key_rotation.py
```

### Test Coverage
- ✅ Default model chat (Path A)
- ✅ Default model streaming (Path A)
- ✅ Specific model chat (Path B)
- ✅ Specific model streaming (Path B)
- ✅ Provider rotation verification
- ✅ API key rotation verification
- ✅ Async chat operations
- ✅ Async streaming operations
- ✅ Multi-turn conversations
- ✅ Multiple provider support

## Components

### APIKeyManager (`ChatServer/llm/api_key_manager.py`)
- Manages API key rotation per provider
- Thread-safe round-robin key selection
- Parses comma-separated environment variables
- Caches parsed keys

### ProviderConfig (`ChatServer/services/chat_service.py`)
- Updated structure:
  - `provider`: LLMProvider enum
  - `default_model`: Default model name
  - `api_keys_env`: Environment variable name (comma-separated)
  - `get_api_keys()`: Returns list of API keys

### ChatService (`ChatServer/services/chat_service.py`)
- Main service class with updated cache structure
- Cache key: `(provider, key_index)` instead of just `provider`
- New methods:
  - `_create_llm_with_key()`: Create on-demand instances
  - `_chat_with_specific_model()`: Path B for non-streaming
  - `_chat_stream_with_specific_model()`: Path B for streaming
  - `_chat_async_with_specific_model()`: Path B for async
  - `_chat_stream_async_with_specific_model()`: Path B for async streaming

### API Schemas (`Server/models/schemas.py`)
- Added `model` field to `ChatRequest`
- Optional field, None = Path A, specified = Path B

### API Routes (`Server/routes/chat.py`)
- Updated all endpoints to pass `model` parameter
- Added `AllKeysFailedError` handling
- Endpoints: `/api/chat`, `/api/chat/stream`, `/api/chat/ws`

## Monitoring & Logging

### Log Levels
- **INFO**: Successful requests with provider/model info
- **WARNING**: Individual key failures with retry indication
- **ERROR**: All keys/providers failed

### Example Logs
```
INFO: Successfully served request using groq with model llama-3.3-70b-versatile (key index 0)
WARNING: Key 1 for groq failed: Rate limit exceeded. Trying next key...
ERROR: All 3 keys failed for groq with model llama-3.1-8b-instant
```

## Performance Considerations

### Memory Usage
- **Cached instances**: ~6 LLM instances (~10MB each = 60MB total)
- **On-demand instances**: Created/destroyed per request (no persistent memory)

### Request Latency
- **Path A (cached)**: ~50ms (instance reuse)
- **Path B (on-demand)**: ~150ms (instance creation + request)
- **First chunk (streaming)**: +50ms (connection test)

### Rate Limit Management
- **Provider rotation**: Distributes load across providers
- **Key rotation**: Distributes load across API keys
- **Fallback**: Automatic retry with different keys/providers

## Available Models

### Groq
- `llama-3.3-70b-versatile` (default)
- `llama-3.1-8b-instant`
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### Cerebras
- `llama-3.3-70b` (default)
- `llama-3.1-8b`
- `llama-3.1-70b`

## Migration from Single-Key System

### Before (Single Key)
```python
# .env
GROQ_API_KEY="gsk_abc123"
CEREBRAS_API_KEY="csk_xyz789"

# Code
response = chat_inference(session)
```

### After (Multi-Key)
```python
# .env
GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3"
CEREBRAS_API_KEYS="csk_key1,csk_key2,csk_key3"

# Code (same API, backward compatible)
response = chat_inference(session)

# New capability: specific model
response = chat_inference(session, model="llama-3.1-8b-instant")
```

## Troubleshooting

### Issue: "No API keys configured"
**Solution**: Check `.env` file has `GROQ_API_KEYS` or `CEREBRAS_API_KEYS` set

### Issue: AllKeysFailedError
**Solution**: 
1. Verify API keys are valid
2. Check rate limits on provider dashboard
3. Try with `fallback=False` to see specific error

### Issue: Model not found
**Solution**: Check model name spelling, ensure provider supports that model

### Issue: Cache not working
**Solution**: Cache only works for default models (Path A), not specific models (Path B)
