from functools import lru_cache
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def create_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model.
    """

    model = SentenceTransformer(EMBEDDING_MODEL)

    return model