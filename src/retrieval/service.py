import logging
from typing import Any, Dict, List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from src.infra.llm.local import OllamaLLM
from src.infra.vectorstore.base import BaseVectorStore
from src.infra.vectorstore.chroma import ChromaVectorStore
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


def _chromadb_to_metadata_filters(
    filters_dict: Dict[str, Any],
) -> Optional[MetadataFilters]:
    """
    Convert a ChromaDB-style filter dict (from QueryAnalyzer) to a
    LlamaIndex MetadataFilters object.

    Handles both single-field and $and multi-field formats:
      {"filename": "report.pdf"}
      {"$and": [{"filename": "report.pdf"}, {"section": "Safety"}]}
    """
    if not filters_dict:
        return None

    filter_list: List[MetadataFilter] = []

    if "$and" in filters_dict:
        for sub in filters_dict["$and"]:
            for k, v in sub.items():
                filter_list.append(MetadataFilter(key=k, value=str(v)))
    else:
        for k, v in filters_dict.items():
            if not k.startswith("$"):
                filter_list.append(MetadataFilter(key=k, value=str(v)))

    return MetadataFilters(filters=filter_list) if filter_list else None


def _node_to_dict(node: NodeWithScore) -> Dict[str, Any]:
    """Convert a LlamaIndex NodeWithScore to the dict format expected by the orchestrator."""
    return {
        "text": node.node.get_content(),
        "metadata": node.node.metadata,
        "score": node.score or 0.0,
        "id": node.node.node_id,
    }


class RetrievalService:
    def __init__(
        self,
        llm: OllamaLLM,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """
        Args:
            llm:          Ollama LLM used by QueryAnalyzer for filter extraction.
            vector_store: Vector store backend. Defaults to ChromaVectorStore.
                          Pass any BaseVectorStore subclass to swap backends.
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.query_analyzer = QueryAnalyzer(llm)
        self.reranker = Reranker()

    def retrieve(
        self, query: str, limit: int = 20, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline: Analyze → Search → Rerank.

        1. QueryAnalyzer extracts metadata filters from the query via LLM.
        2. VectorIndexRetriever fetches `limit` candidate nodes from ChromaDB.
        3. SentenceTransformerRerank scores and returns top `top_k` nodes.
        4. Results are converted to dicts for the orchestrator.
        """
        # 1. Extract metadata filters (LLM-powered)
        filters_dict = self.query_analyzer.analyze(query)
        lm_filters = _chromadb_to_metadata_filters(filters_dict)

        if lm_filters:
            logger.debug(f"Applying metadata filters: {filters_dict}")

        # 2. Retrieve candidates
        retriever = self.vector_store.as_retriever(
            similarity_top_k=limit,
            filters=lm_filters,
        )
        candidates: List[NodeWithScore] = retriever.retrieve(query)
        logger.debug(f"Retrieved {len(candidates)} candidates from vector store.")

        # 3. Rerank
        final_nodes = self.reranker.rerank(query, candidates, top_k=top_k)
        logger.debug(f"Reranked to top {len(final_nodes)} nodes.")

        # 4. Convert to dicts for orchestrator
        return [_node_to_dict(n) for n in final_nodes]

    def add_nodes(self, nodes: List[Any]) -> None:
        """Persist nodes (List[TextNode]) to the vector store."""
        self.vector_store.add_nodes(nodes)

    def delete_document(self, document_id: int) -> None:
        """Remove all nodes for the given document from the vector store."""
        self.vector_store.delete_document(document_id)
