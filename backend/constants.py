"""Application-wide constants."""

# SSE Streaming
SSE_CHUNK_SIZE = 32

# RAG Relevance Scoring Weights
RELEVANCE_SIMILARITY_WEIGHT = 0.4
RELEVANCE_POSITION_WEIGHT = 0.3
RELEVANCE_CHUNK_COUNT_WEIGHT = 0.3

# Position Decay for Relevance Calculation
POSITION_DECAY_FACTOR = 0.95

# Database Connection Pool
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10

# File Upload
MAX_UPLOAD_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
