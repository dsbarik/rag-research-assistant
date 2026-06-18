from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config.settings import EMBEDDING_MODEL


def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Returns a LlamaIndex HuggingFaceEmbedding using the model defined in settings.

    Model: BAAI/bge-large-en-v1.5
      - 1024-dim dense embeddings
      - Top MTEB performance for retrieval tasks
      - Strong on scientific and technical text
      - No trust_remote_code required
    """
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        embed_batch_size=32,
        device="cpu",
    )
