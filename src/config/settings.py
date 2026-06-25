from pathlib import Path

import torch

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main Data Directory
DATA_DIR = PROJECT_ROOT / "data"

# Raw Documents Directory
DOCUMENTS_DIR = DATA_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Vector Store Directory
# Embedding model — BAAI/bge-large-en-v1.5
# Top MTEB performer, 1024-dim, no trust_remote_code needed, strong on scientific text
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Database file
DATABASE_PATH = DATA_DIR / "documents.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# LLM
LLM_MODEL_NAME = "gemma4:31b-cloud"

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
