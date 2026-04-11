import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dialogue import router as dialogue_router
from api.folder import router as folder_router
from api.message import router as message_router
from api.source import router as source_router
from api.user import router as user_router
from db import close_db, init_db
from settings import settings

# Add rag module to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "rag"))

from rag import Settings as RagSettings
from services.rag_service import init_rag_service, shutdown_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    print("Starting Knowledge Base RAG Backend...")

    # Initialize database
    await init_db()
    print("✓ Database initialized")

    # Initialize RAG service if enabled
    if settings.RAG_ENABLED:
        try:
            # Create RAG settings
            rag_settings = RagSettings(
                embedding_model=settings.RAG_EMBEDDING_MODEL,
                rerank_model=settings.RAG_RERANK_MODEL,
                llm_model_generation=settings.RAG_LLM_MODEL,
                llm_api_url=settings.RAG_LLM_API_URL,
                llm_api_key=settings.RAG_LLM_API_KEY,
                top_k=settings.RAG_TOP_K,
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                chroma_path=settings.RAG_CHROMA_PATH,
                chroma_collection=settings.RAG_CHROMA_COLLECTION,
            )

            # Initialize RAG service
            classifier_path = Path(settings.OUTLIER_CLASSIFIER_PATH) if settings.OUTLIER_DETECTION_ENABLED else None
            init_rag_service(
                rag_settings=rag_settings,
                classifier_path=classifier_path,
                enable_outlier_detection=settings.OUTLIER_DETECTION_ENABLED,
            )
            print("✓ RAG service initialized")
        except Exception as e:
            print(f"⚠ Failed to initialize RAG service: {e}")
            print("  Backend will fallback to external RAG API")

    print("✓ Backend started successfully")

    yield

    # Shutdown
    print("Shutting down Knowledge Base RAG Backend...")
    await close_db()
    if settings.RAG_ENABLED:
        shutdown_rag_service()
    print("✓ Backend shutdown complete")


app = FastAPI(
    docs_url="/api/docs",
    title="Knowledge Base RAG Backend API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(dialogue_router)
app.include_router(message_router)
app.include_router(source_router)
app.include_router(folder_router)
