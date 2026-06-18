import gc
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlmodel import SQLModel

from src.infra.db.session import engine
from src.utils.logging import setup_logging

from .router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    SQLModel.metadata.create_all(engine)
    yield

    gc.collect()


app = FastAPI(
    title="RAG System API",
    description="RAG with vector search and KG verification",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
