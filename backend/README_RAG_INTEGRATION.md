# RAG Integration Guide

This document explains how the RAG (Retrieval-Augmented Generation) and Outlier Detection modules are integrated into the backend API.

## Architecture

The backend now includes:
- **RAG Service**: Local RAG system for answering questions with context retrieval
- **Outlier Detection**: Topic classifier to filter off-topic questions
- **Fallback**: External RAG API fallback if local RAG fails

## Components

### 1. RAG Service (`services/rag_service.py`)

Provides:
- `RagService` class for managing RAG models
- `answer_question()` - Main method for answering questions
- `retrieve_chunks()` - Retrieval with configurable strategies
- `check_topic()` - Outlier detection

**Retrieval Strategies:**
- `basic` - Simple vector similarity (~5ms)
- `rerank` - With cross-encoder reranking (~50ms) **[DEFAULT]**
- `query_transform` - Query rewriting + HyDE (~200ms)

### 2. Settings (`settings.py`)

New configuration options:

```python
# RAG Settings
RAG_ENABLED=true                          # Enable/disable local RAG
RAG_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 # Embedding model
RAG_RERANK_MODEL=BAAI/bge-reranker-base   # Reranker model
RAG_LLM_MODEL=TechxGenus/c4ai-command-r-v01-AWQ  # Generation model
RAG_LLM_API_URL=http://localhost:8003/v1  # LLM API endpoint
RAG_LLM_API_KEY=                          # LLM API key (if needed)
RAG_TOP_K=5                               # Number of retrieved chunks
RAG_CHUNK_SIZE=1000                       # Document chunk size
RAG_CHUNK_OVERLAP=200                     # Overlap between chunks
RAG_CHROMA_PATH=../data/chromadb          # ChromaDB storage path
RAG_CHROMA_COLLECTION=docs_fast           # Collection name

# Outlier Detection Settings
OUTLIER_DETECTION_ENABLED=true                    # Enable topic checking
OUTLIER_CLASSIFIER_PATH=./models/pytorch_classifier.joblib  # Classifier model
OUTLIER_REJECT_OFF_TOPIC=false                    # Auto-reject off-topic questions

# Code Executor Settings
CODE_EXECUTOR_URL=http://localhost:8002/execute   # Code executor endpoint
```

### 3. Application Lifespan (`app.py`)

Models are loaded during startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Initialize RAG service
    init_rag_service(...)

    yield

    # Shutdown - Cleanup resources
    shutdown_rag_service()
```

### 4. Message Controller (`api/message/controller.py`)

Flow:
1. Try local RAG service (if enabled)
2. Fallback to external RAG API (if local fails)
3. Extract sources from retrieved chunks
4. Parse and execute code blocks
5. Return response with sources and code execution results

## Setup Instructions

### 1. Install Dependencies

The RAG and outlier-detection modules have their own dependencies:

```bash
# Install RAG dependencies
cd ../rag
pip install -e .

# Install outlier-detection dependencies
cd ../outlier-detection
pip install -e .
```

### 2. Prepare ChromaDB Collection

You need to index your documents into ChromaDB first:

```python
from rag import Settings, create_embed_model
from rag.documents import load_documents, split_documents
from rag.vectorstore import create_collection, index_chunks

# Load settings
settings = Settings()

# Load and split documents
docs = load_documents(settings.dataset_path)  # Load .md files
chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

# Create embedding model
embed_model = create_embed_model(settings)

# Create ChromaDB collection
collection = create_collection(settings.chroma_path, settings.chroma_collection)

# Index chunks
index_chunks(collection, chunks, embed_model)
```

Or use the existing collection from the root `data/chromadb/` directory.

### 3. (Optional) Train Topic Classifier

If you want outlier detection:

```python
from outlier_detection import TopicClassifier

# Prepare training texts (PyTorch-related questions)
pytorch_texts = [
    "How to create a tensor in PyTorch?",
    "What is torch.nn.Module?",
    "How to use DataLoader?",
    # ... more examples (50-200 recommended)
]

# Train classifier
classifier = TopicClassifier(nu=0.05)  # 5% outliers
classifier.fit(pytorch_texts)

# Save model
classifier.save("backend/models/pytorch_classifier.joblib")
```

### 4. Setup LLM API

You need a running LLM API endpoint compatible with OpenAI API format:

**Option A: Use vLLM**
```bash
vllm serve TechxGenus/c4ai-command-r-v01-AWQ \
  --port 8003 \
  --max-model-len 4096
```

**Option B: Use Ollama**
```bash
ollama serve
# Set RAG_LLM_API_URL=http://localhost:11434/v1
```

**Option C: Use OpenAI API**
```bash
# Set RAG_LLM_API_URL=https://api.openai.com/v1
# Set RAG_LLM_API_KEY=your_openai_api_key
```

### 5. Start Backend

```bash
cd backend
uv run uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

On startup, you should see:
```
Starting Knowledge Base RAG Backend...
✓ Database initialized
Loading RAG models...
✓ Loaded chat model: TechxGenus/c4ai-command-r-v01-AWQ
✓ Loaded embedding model: BAAI/bge-base-en-v1.5
✓ Loaded reranker: BAAI/bge-reranker-base
✓ Loaded ChromaDB collection: docs_fast
✓ Loaded topic classifier from: ./models/pytorch_classifier.joblib
✓ RAG service initialized
✓ Backend started successfully
```

## Usage

### Send Message API

```bash
POST /api/message
Authorization: Bearer <jwt_token>

{
  "dialogue_id": 1,
  "message": "How to create a tensor?"
}
```

**Response:**
```json
{
  "message_id": 123,
  "user_message": "How to create a tensor?",
  "assistant_response": "To create a tensor in PyTorch, use torch.tensor()...",
  "sources": [
    "pytorch/docs/tensor.md",
    "pytorch/docs/creation_ops.md"
  ],
  "code_executions": [
    {
      "code": "import torch\nx = torch.tensor([1, 2, 3])\nprint(x)",
      "success": true,
      "stdout": "tensor([1, 2, 3])\n",
      "stderr": "",
      "result": null,
      "error": null
    }
  ],
  "created_at": "2026-04-11T..."
}
```

## Fallback Behavior

If local RAG service is unavailable, the system automatically falls back to the external RAG API at `http://localhost:8000/forward`.

Fallback triggers:
1. `RAG_ENABLED=false` in settings
2. RAG service initialization failed
3. Runtime error during RAG processing

## Performance

### Latency (single query, GPU-accelerated):
- **Basic retrieval**: ~50ms
- **Rerank retrieval**: ~150ms (default)
- **Query transform retrieval**: ~500ms
- **LLM generation**: 500-2000ms (depends on model and output length)

**Total latency**: ~700-2500ms per question

### Memory Requirements:
- **Embedding model**: ~500MB
- **Reranker model**: ~700MB
- **ChromaDB**: Varies by document count (~1GB for 10k chunks)
- **LLM**: Depends on model (4-32GB)

### Optimization Tips:
1. Use GPU for faster inference
2. Use smaller embedding models for lower memory
3. Reduce `RAG_TOP_K` for faster retrieval
4. Use `basic` or `rerank` strategy instead of `query_transform`
5. Cache frequently asked questions

## Troubleshooting

### Issue: "RAG service not initialized"

**Solution:**
- Check that ChromaDB collection exists
- Verify LLM API is running and accessible
- Check logs for initialization errors

### Issue: Slow responses

**Solution:**
- Reduce `RAG_TOP_K` value
- Use `basic` or `rerank` strategy
- Enable GPU acceleration
- Check LLM API performance

### Issue: Off-topic questions not filtered

**Solution:**
- Train topic classifier with more diverse examples
- Adjust `nu` parameter (lower = stricter)
- Set `OUTLIER_REJECT_OFF_TOPIC=true` for auto-rejection

### Issue: Poor answer quality

**Solution:**
- Increase `RAG_TOP_K` for more context
- Use `query_transform` strategy
- Check if documents are properly indexed
- Verify LLM model is appropriate for the task

## Development

### Testing RAG Service

```python
from services.rag_service import get_rag_service

# Get service
rag = get_rag_service()

# Test question
response = rag.answer_question("How to create a tensor?")

print(f"Answer: {response.answer}")
print(f"On topic: {response.is_on_topic} (confidence: {response.topic_confidence})")
print(f"Sources: {[c.source for c in response.chunks]}")
```

### Monitoring

Key metrics to track:
- Response latency
- Retrieval quality (chunk relevance)
- Off-topic question rate
- Fallback usage rate
- User feedback (like/dislike)

## References

- RAG module: `../rag/`
- Outlier-detection module: `../outlier-detection/`
- RAG service: `services/rag_service.py`
- Message controller: `api/message/controller.py`
