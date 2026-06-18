import logging
from typing import List, Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore as _LlamaChromaVS

from src.config.settings import VECTOR_STORE_DIR
from src.infra.embeddings.sentence_transformer import get_embedding_model
from src.infra.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = VECTOR_STORE_DIR / "chroma_db"


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store using LlamaIndex as the indexing layer.

    Internally uses:
      - llama_index.vector_stores.chroma.ChromaVectorStore  (storage)
      - llama_index.core.VectorStoreIndex                   (embed + index + retrieve)
      - HuggingFaceEmbedding (BAAI/bge-large-en-v1.5)       (embedding model)

    Swap backends: subclass BaseVectorStore and implement add_nodes,
    as_retriever, and delete_document.
    """

    def __init__(self, collection_name: str = "rag_collection") -> None:
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

        # 1. Raw ChromaDB client (persistent)
        self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            collection_name
        )

        # 2. LlamaIndex ChromaVectorStore wrapping the collection
        llama_vs = _LlamaChromaVS(chroma_collection=self._chroma_collection)

        # 3. StorageContext wires the vector store into LlamaIndex
        storage_ctx = StorageContext.from_defaults(vector_store=llama_vs)

        # 4. Embedding model (BAAI/bge-large-en-v1.5)
        embed_model = get_embedding_model()

        # 5. VectorStoreIndex — the main interface for insert + query
        self.index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=False,
        )

    # ------------------------------------------------------------------ #
    #  BaseVectorStore interface                                           #
    # ------------------------------------------------------------------ #

    def add_nodes(self, nodes: List[BaseNode]) -> None:
        """Embed and persist nodes into ChromaDB via LlamaIndex."""
        if not nodes:
            return
        self.index.insert_nodes(nodes)
        logger.info(f"Persisted {len(nodes)} nodes to ChromaDB.")

    def as_retriever(
        self,
        similarity_top_k: int = 20,
        filters: Optional[MetadataFilters] = None,
    ) -> BaseRetriever:
        """Return a VectorIndexRetriever configured with optional metadata filters."""
        return self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )

    def delete_document(self, document_id: int) -> None:
        """
        Delete all nodes belonging to a document.
        Uses ChromaDB's native `where` filter on the 'document_id' metadata field.
        """
        self._chroma_collection.delete(
            where={"document_id": str(document_id)}
        )
        logger.info(f"Deleted all nodes for document_id={document_id} from ChromaDB.")
