from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.embedding_model import EMBEDDING_MODEL
from src.vectorstore.chroma_store import (
    LocalEmbeddingFunction,
    COLLECTION_NAME,
)


PERSIST_DIRECTORY = Path("data/vectorstore")


# Load the embedding function once.
# This avoids loading the Sentence Transformer model
# every time we perform an MMR search.
embedding_function = LocalEmbeddingFunction(
    EMBEDDING_MODEL
)


# Load the existing persistent Chroma database once.
# We are using the same vector database as semantic search.
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embedding_function,
)


def retrieve_mmr_documents(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
):
    """
    Retrieve documents using Maximal Marginal Relevance (MMR).

    MMR balances:
    1. Relevance to the user's query.
    2. Diversity among the retrieved documents.
    """

    results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    return results