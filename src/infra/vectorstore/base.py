from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import BaseNode, NodeWithScore
    from llama_index.core.vector_stores import MetadataFilters


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store backends.

    All concrete implementations (Chroma, FAISS, Qdrant, etc.) must satisfy
    this interface, keeping the rest of the codebase decoupled from any
    specific vector store library.

    Data model: LlamaIndex TextNode / NodeWithScore objects are the native
    currency at this layer. The retrieval service is responsible for converting
    to/from the dict format consumed by the orchestrator.
    """

    @abstractmethod
    def add_nodes(self, nodes: List["BaseNode"]) -> None:
        """
        Embed and persist a list of LlamaIndex nodes.

        Args:
            nodes: TextNode objects produced by the document ingestion pipeline.
                   Each node must have 'document_id' set in its metadata.
        """

    @abstractmethod
    def as_retriever(
        self,
        similarity_top_k: int = 20,
        filters: Optional["MetadataFilters"] = None,
    ) -> "BaseRetriever":
        """
        Return a configured retriever for this backend.

        Args:
            similarity_top_k: Number of candidate nodes to fetch before reranking.
            filters:          Optional LlamaIndex MetadataFilters for pre-filtering.

        Returns:
            A LlamaIndex BaseRetriever whose .retrieve(query) returns List[NodeWithScore].
        """

    @abstractmethod
    def delete_document(self, document_id: int) -> None:
        """
        Remove all nodes belonging to the given document.

        Args:
            document_id: The integer ID of the SQLModel Document record.
                         Nodes are matched via the 'document_id' metadata field.
        """
