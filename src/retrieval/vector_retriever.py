from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.embedding_model import EMBEDDING_MODEL
from src.vectorstore.chroma_store import (
    LocalEmbeddingFunction,
    COLLECTION_NAME,#So this tells the retriever which Chroma collection to access.
)#Import our embedding adapter and collection name


PERSIST_DIRECTORY = Path("data/vectorstore")#Our existing Chroma database is located here.


# Load the embedding function once
embedding_function = LocalEmbeddingFunction(
    EMBEDDING_MODEL
)


# Load the existing vector store once
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,#Use the rag_documents collection.
    persist_directory=str(PERSIST_DIRECTORY),#str() converts the Path object into a normal string
    embedding_function=embedding_function,#Use this embedding model when converting queries into vectors.
)


def load_vectorstore() -> Chroma:#The -> Chroma means:This function is expected to return a Chroma object.
    """
    Load the existing persistent Chroma vector store.
    """

    return vectorstore


def retrieve_documents(
    query: str,
    k: int = 5,
):
    """
    Retrieve the top-k most similar chunks for a query,
    along with their similarity scores.
    """

    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )#Find the k chunks that are most similar to this query and give me their scores.

    return results