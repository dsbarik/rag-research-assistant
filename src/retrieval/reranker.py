from typing import List

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle


class Reranker:
    """
    Cross-encoder reranker using LlamaIndex's SentenceTransformerRerank.

    Replaces the previous raw CrossEncoder implementation with LlamaIndex's
    native postprocessor, which integrates cleanly with NodeWithScore objects.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO (passage retrieval)
      - Lightweight and fast on CPU/MPS
      - Strong out-of-the-box for scientific and technical text
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        self._reranker = SentenceTransformerRerank(
            model=model_name,
            top_n=top_n,
        )

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 5,
    ) -> List[NodeWithScore]:
        """
        Rerank a list of retrieved nodes by cross-encoder relevance score.

        Args:
            query:  The original user query string.
            nodes:  Candidate nodes from the vector retriever.
            top_k:  Number of top nodes to return after reranking.

        Returns:
            Top-k NodeWithScore objects, sorted by descending rerank score.
        """
        if not nodes:
            return []
        self._reranker.top_n = top_k
        return self._reranker.postprocess_nodes(
            nodes, query_bundle=QueryBundle(query_str=query)
        )
