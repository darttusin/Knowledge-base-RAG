from fastapi import FastAPI
from generation import rag_answer
from pydantic import BaseModel
from retrival import chromadb_deduplication_search

app = FastAPI(docs_url="/api/docs")


class QueryResponse(BaseModel):
    response: str


@app.get("/api/query")
async def query(text: str) -> QueryResponse:
    return QueryResponse(
        response=rag_answer(text, chromadb_deduplication_search(query=text))
    )
