# Quick Start Guide - Multi-API Key Rotation

Get started with the multi-API key rotation system in 5 minutes.

## Step 1: Configure Environment

Create or update your `.env` file:

```bash
# backend/.env
GROQ_API_KEYS="your_key1,your_key2,your_key3"
CEREBRAS_API_KEYS="your_key1,your_key2,your_key3"
```

**Note**: You can have 1 or more keys per provider. Keys are comma-separated.

## Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## Step 3: Test the System

Run the comprehensive test suite:

```bash
python tests/test_multi_key_rotation.py
```

Expected output:
```
✅ Test 1: Default Model Chat (Path A)
✅ Test 2: Default Model Streaming (Path A)
✅ Test 3: Specific Model Chat (Path B)
✅ Test 4: Specific Model Streaming (Path B)
... (12 tests total)
```

## Step 4: Start the Server

```bash
uvicorn Server.main:app --reload --host 0.0.0.0 --port 8000
```

## Step 5: Make Your First Request

### Option A: Default Model (Automatic Rotation)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "fallback": true
  }'
```

### Option B: Specific Model (Key-Level Fallback)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "model": "llama-3.1-8b-instant",
    "provider": "groq",
    "fallback": true
  }'
```

## Using Python SDK

```python
from ChatServer import ChatSession, LLMProvider, chat_inference

# Create a session
session = ChatSession()
session.add_user_message("Hello!")

# Option A: Default model (automatic rotation)
response = chat_inference(session, fallback=True)
print(response.content)

# Option B: Specific model
response = chat_inference(
    session,
    provider=LLMProvider.GROQ,
    model="llama-3.1-8b-instant",
    fallback=True
)
print(response.content)
```

## Streaming Example

```python
from ChatServer import ChatSession, chat_inference_stream

session = ChatSession()
session.add_user_message("Write a short poem")

for chunk in chat_inference_stream(session, fallback=True):
    print(chunk, end="", flush=True)
```

## Key Concepts

### Path A: Default Models (Cached)
- **No model parameter** → Uses default models
- **Provider rotation** → Groq → Cerebras → Groq...
- **Cached instances** → Fast response (~50ms)
- **Use for**: General purpose chat

### Path B: Specific Models (On-Demand)
- **Model parameter specified** → Uses exact model
- **Key rotation** → Tries all API keys for that provider
- **On-demand creation** → Slower (~150ms) but flexible
- **Use for**: When you need specific model capabilities

## Available Models

### Groq
- `llama-3.3-70b-versatile` ⭐ (default)
- `llama-3.1-8b-instant` (fastest)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

### Cerebras
- `llama-3.3-70b` ⭐ (default)
- `llama-3.1-8b`
- `llama-3.1-70b`

## Troubleshooting

### "No API keys configured"
→ Check your `.env` file has `GROQ_API_KEYS` or `CEREBRAS_API_KEYS` set

### AllKeysFailedError
→ Verify API keys are valid and check rate limits on provider dashboard

### Module not found
→ Make sure you're in the `backend/` directory and dependencies are installed

## Next Steps

1. ✅ Read [MULTI_KEY_ROTATION.md](MULTI_KEY_ROTATION.md) for detailed documentation
2. ✅ Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. ✅ Explore [tests/test_multi_key_rotation.py](tests/test_multi_key_rotation.py) for more examples
4. ✅ Monitor server logs to see rotation in action

## Support

- Check logs for detailed error messages
- Review `.env.example` for configuration examples
- Run tests to verify your setup

---

**That's it!** You're now running a production-ready multi-API key rotation system with automatic fallback and load balancing. 🚀
