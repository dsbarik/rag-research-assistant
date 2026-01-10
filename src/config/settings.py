from pathlib import Path


# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main Data Directory
DATA_DIR = PROJECT_ROOT / "data"

# Raw Documents Directory
DOCUMENTS_DIR = DATA_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Vector Store Directory
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Database file
DATABASE_PATH = DATA_DIR / "documents.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# LLM
LLM_MODEL_NAME = "llama3.2"