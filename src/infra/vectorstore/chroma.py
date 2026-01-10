import logging
from typing import List, Dict, Any

import chromadb

from src.config import VECTOR_STORE_DIR
from src.infra.embeddings.sentence_transformer import get_embedding_function

CHROMA_DB_PATH = VECTOR_STORE_DIR / "chroma_db"


class ChromaVectorStore:

    def __init__(self, collection_name: str = "rag_collection"):
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

        # Inject the embedding function from the separate module
        self.embedding_fn = get_embedding_function()

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])

            # Metadata sanitization
            meta = chunk.get("metadata", {}).copy()
            meta["document_id"] = chunk["document_id"]

            clean_meta = {}
            for k, v in meta.items():
                if v is not None:
                    # Chroma only accepts str, int, float, bool
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)

            metadatas.append(clean_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logging.info(f"Persisted {len(chunks)} chunks to ChromaDB.")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:

        results = self.collection.query(query_texts=[query], n_results=limit)

        parsed_results = []

        if results["ids"]:
            num_results = len(results["ids"][0])
            for i in range(num_results):
                parsed_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i], # type: ignore
                        "metadata": results["metadatas"][0][i], # type: ignore
                        "score": (
                            results["distances"][0][i] if results["distances"] else 0.0
                        ),
                    }
                )

        return parsed_results

    def delete_document(self, document_id: int):
        self.collection.delete(where={"document_id": str(document_id)},)
