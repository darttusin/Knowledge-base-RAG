from db import close_db, init_db
from fastapi import FastAPI
from routes.auth import router as auth_router
from routes.forward import router as forward_router
from routes.history import router as history_router
from routes.stats import router as stats_router

app = FastAPI(docs_url="/api/docs")

app.add_event_handler("startup", init_db)
app.add_event_handler("shutdown", close_db)

app.include_router(forward_router)
app.include_router(history_router)
app.include_router(stats_router)
app.include_router(auth_router)
