from contextlib import asynccontextmanager
import shutil
import gc

from fastapi import FastAPI
from sqlmodel import SQLModel

from src.infra.db.session import engine
from .router import router
from src.config import DATA_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield
    
    gc.collect()
    
    shutil.rmtree(DATA_DIR, ignore_errors=True)


app = FastAPI(
    title="RAG System API",
    description="RAG with vector search and KG verification",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")