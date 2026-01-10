from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions


class SentenceTransformerEmbeddings(EmbeddingFunction):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._impl = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    def __call__(self, documents: Documents) -> Embeddings:
        return self._impl(documents)


def get_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingFunction:
    return SentenceTransformerEmbeddings(model_name=model_name)
