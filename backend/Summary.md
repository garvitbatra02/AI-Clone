# ChatClone Backend — Architecture & Summary

A modular, production-ready AI backend providing multi-provider LLM chat, retrieval-augmented generation (RAG), and an asset upload pipeline — all served through a single FastAPI application.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Module Overview](#module-overview)
- [ChatService](#chatservice)
- [RAGService](#ragservice)
- [AssetUploadService](#assetuploadservice)
- [Shared Infrastructure](#shared-infrastructure)
- [API Endpoints](#api-endpoints)
- [Request Flows](#request-flows)
- [Environment Variables](#environment-variables)
- [Tech Stack](#tech-stack)
- [Key Design Patterns](#key-design-patterns)

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                         │
│                     (ChatService/Server/main.py)                   │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Chat Routes  │  │  RAG Routes  │  │  Asset Upload Routes     │ │
│  │  /api/chat    │  │  /api/rag    │  │  /api/assets/collections │ │
│  │  /api/chat/ws │  │              │  │  /api/assets/uploads     │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘ │
└─────────┼─────────────────┼───────────────────────┼───────────────┘
          │                 │                       │
          ▼                 ▼                       ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────────────────┐
│   ChatService   │ │  RAGService │ │    AssetUploadService        │
│                 │ │             │ │    (DashboardService)         │
│ Provider        │ │ Retrieval → │ │                               │
│ Rotation:       │ │ Rerank →    │ │ Load → Chunk → Embed → Store │
│ Groq→Cerebras→  │ │ Generate    │ │                               │
│ Cohere          │ │             │ │                               │
└────────┬────────┘ └──────┬──────┘ └──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Core Infrastructure                          │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐│
│  │  LLM Layer   │  │  Embeddings  │  │       VectorDB            ││
│  │  Groq        │  │  Cohere      │  │       Qdrant              ││
│  │  Cerebras    │  │  embed-v3.0  │  │  (Cloud / In-Memory)      ││
│  │  Cohere      │  │              │  │                           ││
│  └──────────────┘  └──────────────┘  └───────────────────────────┘│
│                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  KeyRotationMixin — Multi-key rotation, caching, retry logic  ││
│  │  (shared/key_rotation.py — inherited by all API services)     ││
│  └────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

---

## Module Overview

```
backend/
├── ChatService/              # LLM chat + server entry point
│   ├── Chat/
│   │   ├── llm/              # LLM abstraction layer
│   │   │   ├── base.py           # BaseLLM, LLMProvider, LLMConfig, LLMResponse
│   │   │   ├── factory.py        # LLMFactory (create by provider or model name)
│   │   │   ├── model_registry.py # 47+ models → provider mapping
│   │   │   └── proprietary_llms/ # Groq, Cerebras, Cohere implementations
│   │   ├── services/
│   │   │   └── chat_service.py   # Provider rotation + fallback orchestration
│   │   └── session/
│   │       └── chat_session.py   # Message management, conversation history
│   └── Server/
│       ├── main.py               # FastAPI app, lifespan, CORS, router mounting
│       ├── routes/
│       │   ├── chat.py           # /api/chat endpoints (REST + SSE + WebSocket)
│       │   └── rag.py            # /api/rag endpoints
│       └── models/
│           └── schemas.py        # Pydantic request/response schemas
│
├── RAGService/               # Retrieval-Augmented Generation pipeline
│   └── Data/
│       ├── DocumentProcessors/
│       │   ├── base.py           # ProcessedDocument, SupportedFileType, TextSplitter
│       │   ├── loader_factory.py # Auto-detect & load any supported file type
│       │   ├── smart_chunker.py  # Routes files to format-specific strategies
│       │   ├── strategies/       # SemanticChunker, TextChunker, RowChunker, etc.
│       │   └── loaders/          # TextLoader, PDFLoader, CSVLoader, JSONLoader, etc.
│       ├── Embeddings/
│       │   ├── base.py           # BaseEmbeddings, EmbeddingConfig, EmbeddingProvider
│       │   ├── factory.py        # EmbeddingsFactory
│       │   └── providers/        # CohereEmbeddings (embed-english-v3.0)
│       ├── Reranker/
│       │   ├── base.py           # BaseReranker
│       │   └── cohere_reranker.py# CohereReranker (rerank-v3.5)
│       ├── VectorDB/
│       │   ├── base.py           # DocumentChunk, SearchResult, BaseVectorDB
│       │   └── providers/        # QdrantVectorDB (cloud + in-memory)
│       └── services/
│           ├── rag_service.py        # Full RAG pipeline (retrieve → rerank → generate)
│           ├── retrieval_service.py  # Vector search + reranking
│           ├── vectordb_service.py   # High-level VectorDB + embedding ops
│           └── asset_upload_service.py # File upload pipeline (load → chunk → embed → store)
│
├── AssetUploadService/       # Dashboard/upload HTTP layer
│   ├── Server/
│   │   ├── main.py               # Standalone app + get_asset_routers()
│   │   ├── routes/
│   │   │   ├── collections.py    # Collection CRUD endpoints
│   │   │   └── uploads.py        # Upload, preview, text endpoints
│   │   └── models/
│   │       └── schemas.py        # 14 Pydantic schemas
│   └── services/
│       └── dashboard_service.py  # Orchestrates temp-file lifecycle for HTTP uploads
│
├── shared/
│   └── key_rotation.py       # KeyRotationMixin — multi-key rotation for all services
│
├── prompts/
│   └── rag.py                # RAG system prompt template + context formatting
│
├── tests/                    # E2E + unit tests
└── examples/                 # Smoke tests and demos
```

---

## ChatService

### LLM Providers

Three LLM providers are fully implemented, each inheriting `BaseLLM` + `KeyRotationMixin`:

| Provider | Class | Models | SDK | Env Var |
|----------|-------|--------|-----|---------|
| **Groq** | `GroqLLM` | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b, qwen3-32b, + more | `langchain-groq` | `GROQ_API_KEYS` |
| **Cerebras** | `CerebrasLLM` | llama3.1-8b, gpt-oss-120b | `langchain-cerebras` | `CEREBRAS_API_KEYS` |
| **Cohere** | `CohereLLM` | command-a-03-2025, command-r7b-12-2024, command-r-plus, + more | `cohere` SDK | `COHERE_API_KEYS` |

**Model Registry** maps 47+ model names to their providers, enabling auto-detection:
```python
llm = LLMFactory.from_model("llama-3.3-70b-versatile")  # auto-detects → Groq
```

### Provider Rotation & Fallback

`ChatService` provides **two levels of resilience**:

1. **Key rotation** (within a provider): If key 1 fails, try key 2, key 3, etc. Handled by `KeyRotationMixin`.
2. **Provider fallback** (across providers): If Groq is down entirely, fall back to Cerebras, then Cohere.

**Chat priority order:** Groq → Cerebras → Cohere (optimized for speed)
**RAG priority order:** Cohere → Cerebras → Groq (optimized for grounded generation)

Round-robin rotation distributes load across providers. Each call picks the next provider in rotation, with fallback trying all remaining providers on failure.

### Chat Capabilities

| Feature | Method | Description |
|---------|--------|-------------|
| Sync chat | `chat()` | Returns full `LLMResponse` |
| Async chat | `chat_async()` | Awaitable version |
| Streaming | `chat_stream()` | Yields string chunks |
| Async streaming | `chat_stream_async()` | Async iterator of chunks |

### ChatSession

Manages conversation state:
- System prompt (injected at position 0)
- Message history with roles: `SYSTEM`, `USER`, `ASSISTANT`, `FUNCTION`, `TOOL`
- Auto-trimming when `max_history_length` is exceeded
- Each message has a UUID, timestamp, and optional metadata

---

## RAGService

### Document Processing Pipeline

```
File (PDF/TXT/CSV/JSON/DOCX/MD)
  │
  ├─ DocumentLoaderFactory ──→ auto-detects file type, returns ProcessedDocument
  │
  ├─ SmartChunker ──→ routes to format-specific strategy:
  │   ├─ PDF / DOCX  →  SemanticChunker  (heading/section-aware splitting)
  │   ├─ TXT / MD    →  TextChunker      (paragraph/topic-aware splitting)
  │   ├─ CSV         →  RowChunker       (1 row = 1 chunk)
  │   └─ JSON        →  JsonEntryChunker (1 entry = 1 chunk)
  │
  ├─ (Optional) LLM Structural Analysis:
  │   ├─ "structural" mode for PDF/DOCX — detects headings, tables, sections
  │   └─ "topical" mode for TXT/MD — groups lines by topic
  │   Uses Groq/Cerebras via ChatService LLM infrastructure
  │
  └─ Output: List[DocumentChunk] with content, metadata, source, chunk_index
```

**Supported file types:** `.txt`, `.md`, `.markdown`, `.json`, `.csv`, `.pdf`, `.docx`

### Embeddings

| Provider | Model | Dimensions | Env Var |
|----------|-------|------------|---------|
| **Cohere** (default) | embed-english-v3.0 | 1024 | `COHERE_API_KEYS` |

Cohere embeddings use `input_type` to optimize for use case:
- `SEARCH_DOCUMENT` — when embedding stored documents
- `SEARCH_QUERY` — when embedding user queries at search time

### Reranking

| Provider | Model | Description |
|----------|-------|-------------|
| **Cohere** | rerank-v3.5 (multilingual) | Cross-encoder re-scoring of search results |

Reranking narrows vector search candidates (default top-20) to the most relevant results (default top-5). It's optional — if the Cohere API key is missing, reranking is skipped with a warning.

### Vector Database

| Provider | Modes | Env Vars |
|----------|-------|----------|
| **Qdrant** | Cloud (via URL) or In-Memory (for testing) | `QDRANT_URL`, `QDRANT_API_KEY` |

`DocumentChunk` is the unified data model across the entire pipeline — from chunking through embedding to storage and retrieval.

### RAG Pipeline (3-Stage)

```
User Query
  │
  ├── STAGE 1: RETRIEVE
  │   ├── Embed query using Cohere (input_type=SEARCH_QUERY)
  │   ├── Vector search in Qdrant → top-20 candidates
  │   └── Rerank with Cohere reranker → top-5 results
  │
  ├── STAGE 2: AUGMENT
  │   ├── Format retrieved chunks into context string
  │   └── Inject context into RAG system prompt
  │
  └── STAGE 3: GENERATE
      ├── Send augmented prompt to LLM (Cohere → Cerebras → Groq)
      └── Return answer with source citations
```

The RAG system prompt enforces strict grounding rules:
- Only answer from retrieved context, never fabricate
- Admit when information is not available
- Scope-locked to the owner's data

---

## AssetUploadService

The HTTP layer for document management and upload. Can run standalone or mounted into the main app.

### Collection Management
- Create, list, get stats, and delete vector DB collections
- Each collection is a separate namespace in Qdrant

### Upload Pipeline
Two families of endpoints serve different use cases:

| Family | For | Examples |
|--------|-----|---------|
| **Multipart** | Browser / Dashboard | `POST /preview`, `POST /file` |
| **Local-path** | CLI / Testing | `POST /preview/local`, `POST /file/local`, `POST /directory/local` |
| **Text** | Raw text ingestion | `POST /text`, `POST /texts` |

**Preview** (dry-run): Runs the load → chunk pipeline without embedding or storing. Returns file info, total chunks, estimated tokens, and sample chunk previews.

**Upload** (full): Runs the complete load → chunk → embed → store pipeline. Returns document IDs, total chunks, and metadata.

**DashboardService** handles the temp-file lifecycle for HTTP multipart uploads — saves bytes to a temp file (preserving the original extension for type detection), delegates to the core `AssetUploadService`, then cleans up.

---

## Shared Infrastructure

### KeyRotationMixin (`shared/key_rotation.py`)

The backbone of all API resilience. Inherited by `BaseLLM`, `BaseEmbeddings`, and `BaseReranker`.

**What it provides:**
- **API key loading**: Reads comma-separated keys from environment variables (e.g., `COHERE_API_KEYS=key1,key2,key3`)
- **Per-key client caching**: Each API key gets its own cached SDK client instance
- **Rotation loop**: On failure, tries the next key. After exhausting all keys, retries the full sweep (configurable `max_retries`).
- **Stale client detection**: Detects "Event loop is closed" errors from cached async clients and auto-recreates them
- **Streaming support**: Validates the first chunk before committing to a stream

**Rotation behavior (5 keys, max_retries=2):**
```
Sweep 0: Key 0 → Key 1 → Key 2 → Key 3 → Key 4  (first success exits)
Sweep 1: Key 0 → Key 1 → Key 2 → Key 3 → Key 4  (retry if all failed)
Sweep 2: Key 0 → Key 1 → Key 2 → Key 3 → Key 4  (final attempt)
→ AllKeysFailedError if everything fails
```

### RAG Prompt (`prompts/rag.py`)

A carefully crafted system prompt for RAG responses:
- Factual accuracy — only answer from provided context
- Honesty — don't agree just because the user said it
- Scope-locked — refuse questions outside the owner's data
- Warm, professional tone

---

## API Endpoints

### Chat (3 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Synchronous chat with provider rotation |
| `POST` | `/api/chat/stream` | SSE streaming chat |
| `WS` | `/api/chat/ws` | WebSocket real-time chat |

### RAG (4 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rag/chat` | Full RAG: retrieve → rerank → generate |
| `POST` | `/api/rag/chat/stream` | Streaming RAG with SSE |
| `POST` | `/api/rag/search` | Retrieval-only (no LLM generation) |
| `GET` | `/api/rag/collections` | List all VectorDB collections |

### Asset Upload — Collections (4 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/assets/collections` | List all collections with stats |
| `GET` | `/api/assets/collections/{name}` | Get single collection stats |
| `POST` | `/api/assets/collections` | Create a new collection |
| `DELETE` | `/api/assets/collections/{name}` | Delete a collection |

### Asset Upload — Uploads (8 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/assets/uploads/supported-types` | List supported file extensions |
| `POST` | `/api/assets/uploads/preview` | Preview multipart file (dry-run) |
| `POST` | `/api/assets/uploads/file` | Upload multipart file |
| `POST` | `/api/assets/uploads/preview/local` | Preview local file by path |
| `POST` | `/api/assets/uploads/file/local` | Upload local file by path |
| `POST` | `/api/assets/uploads/directory/local` | Upload entire local directory |
| `POST` | `/api/assets/uploads/text` | Upload raw text |
| `POST` | `/api/assets/uploads/texts` | Upload batch of texts |

### System (2 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/` | Service info |

**Total: 21 endpoints** (19 REST + 1 WebSocket + 1 health)

---

## Request Flows

### Chat Flow

```
POST /api/chat { messages, provider?, model? }
  │
  ▼
ChatService.chat_async(session)
  ├── Round-robin provider selection (Groq → Cerebras → Cohere)
  ├── Fallback: try all providers on failure
  │
  ▼
GroqLLM.chat_async(session)               ← selected provider
  ├── KeyRotationMixin: try key 0, 1, 2...
  ├── Groq SDK: self._client.ainvoke(messages)
  │
  ▼
ChatResponse { content, provider, model, tokens }
```

### RAG Flow

```
POST /api/rag/chat { messages, collection_name, top_k, rerank }
  │
  ▼
RAGService.aquery(query, collection)
  │
  ├── 1. RETRIEVE
  │   ├── Cohere Embeddings: embed query (SEARCH_QUERY)
  │   ├── Qdrant: vector search → top-20 candidates
  │   └── Cohere Reranker: rerank → top-5
  │
  ├── 2. AUGMENT
  │   └── Inject retrieved context into RAG system prompt
  │
  ├── 3. GENERATE
  │   └── Isolated ChatService (Cohere → Cerebras → Groq)
  │
  ▼
RAGChatResponse { content, sources, provider, model, tokens }
```

### Asset Upload Flow

```
POST /api/assets/uploads/file (multipart: file + collection_name)
  │
  ▼
DashboardService.upload_uploaded_file()
  ├── Save bytes to temp file (preserve extension)
  │
  ▼
AssetUploadService.async_upload_file()
  ├── 1. Create collection in Qdrant (if needed)
  ├── 2. Load file → ProcessedDocument (via DocumentLoaderFactory)
  ├── 3. Smart-chunk → List[DocumentChunk] (format-specific strategy)
  ├── 4. Embed chunks → Cohere embeddings (SEARCH_DOCUMENT)
  ├── 5. Store in Qdrant → document IDs
  │
  ▼
UploadFileResponse { success, document_ids, total_chunks, metadata }
```

---

## Environment Variables

### Required (at least one LLM + embeddings)

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEYS` | Comma-separated Groq API keys |
| `CEREBRAS_API_KEYS` | Comma-separated Cerebras API keys |
| `COHERE_API_KEYS` | Comma-separated Cohere API keys (LLM + Embeddings + Reranker) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | — | Single Cohere key (backward compatibility fallback) |
| `QDRANT_URL` | — | Qdrant cloud URL (omit for in-memory) |
| `QDRANT_API_KEY` | — | Qdrant cloud API key |
| `DEFAULT_COLLECTION` | `"assets"` | Default VectorDB collection name |
| `VECTORDB_PROVIDER` | `"qdrant"` | Vector database provider |
| `VECTORDB_IN_MEMORY` | `"false"` | Use in-memory VectorDB (for testing) |
| `EMBEDDING_PROVIDER` | `"cohere"` | Embedding model provider |
| `EMBEDDING_MODEL` | — | Override embedding model name |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | Vector search candidates |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | Minimum similarity score |
| `RETRIEVAL_RERANK_ENABLED` | `"true"` | Enable/disable Cohere reranking |
| `RETRIEVAL_RERANK_MODEL` | `"rerank-v3.5"` | Reranker model |
| `RETRIEVAL_RERANK_TOP_N` | `5` | Results after reranking |
| `RAG_TEMPERATURE` | `0.7` | LLM temperature for RAG responses |
| `RAG_MAX_TOKENS` | — | Max tokens for RAG responses |
| `CORS_ORIGINS` | `"http://localhost:3000,http://localhost:5173"` | Allowed CORS origins |
| `HOST` | `"0.0.0.0"` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `RELOAD` | `"true"` | Enable uvicorn hot reload |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI + Uvicorn |
| **LLM SDKs** | langchain-groq, langchain-cerebras, cohere SDK |
| **Embeddings** | Cohere embed-english-v3.0 (1024d) |
| **Reranking** | Cohere rerank-v3.5 |
| **Vector DB** | Qdrant (cloud + in-memory) |
| **Document Loading** | pypdf, python-docx, custom loaders for TXT/CSV/JSON/MD |
| **Schemas** | Pydantic v2 |
| **Real-time** | WebSockets, Server-Sent Events (SSE) |
| **Token counting** | tiktoken |
| **Python** | 3.10+ |

---

## Key Design Patterns

### 1. Two-Tier Resilience
`KeyRotationMixin` handles key rotation **within** a provider (5 Cohere keys rotating). `ChatService` handles rotation **across** providers (Groq → Cerebras → Cohere). Together they provide robust fault tolerance.

### 2. Factory + Registry
LLMs, Embeddings, VectorDB, Document Loaders, and Chunking Strategies all follow the same pattern: **Abstract Base → Registry → Factory → Concrete Implementation**. This makes every component pluggable.

### 3. Singleton Services
All services use lazy singletons (`get_chat_service()`, `get_rag_service()`, etc.) initialized during the FastAPI lifespan. This ensures one-time setup and efficient resource sharing.

### 4. Isolated RAG ChatService
The RAG pipeline creates its **own** `ChatService` instance with a different provider priority (Cohere-first for grounded generation), completely separate from the global chat singleton (Groq-first for speed).

### 5. Smart Dispatch
`SmartChunker` routes each file type to a specialized chunking strategy rather than using one-size-fits-all text splitting. This preserves document structure and produces higher-quality chunks.

### 6. API Key Encapsulation
No caller ever passes, sees, or manages API keys. Each service reads its own keys from environment variables through `KeyRotationMixin`. Keys are comma-separated for multi-key rotation.

### 7. Monolith-First, Microservice-Ready
All services run in a single process but are architecturally separate. `AssetUploadService` can run standalone via `uvicorn AssetUploadService.Server.main:app --port 8001` if needed.
