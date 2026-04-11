from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dialogue import router as dialogue_router
from api.folder import router as folder_router
from api.message import router as message_router
from api.source import router as source_router
from api.user import router as user_router
from db import close_db, init_db

app = FastAPI(docs_url="/api/docs", title="Knowledge Base RAG Backend API")

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

app.add_event_handler("startup", init_db)
app.add_event_handler("shutdown", close_db)

app.include_router(user_router)
app.include_router(dialogue_router)
app.include_router(message_router)
app.include_router(source_router)
app.include_router(folder_router)
