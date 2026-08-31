from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.embedding_model import create_embedding_model


PERSIST_DIRECTORY = Path("data/vectorstore") #Store the Chroma database inside data/vectorstore.
COLLECTION_NAME = "rag_documents" #A collection is basically a group of related vectors/documents inside Chroma.


class LocalEmbeddingFunction: #So this class acts as a bridge/adapter:
    """
    Adapter that allows our SentenceTransformer model
    to work with LangChain/Chroma.
    """

    def __init__(self, model_name: str): #expecting a string containing the model name.
        self.model = create_embedding_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]: #This function receives multiple text chunks
        """
        Convert multiple chunks into embeddings.
        """
        embeddings = self.model.encode(
            texts,#this actually converts the text chunks into embeddings using the SentenceTransformer model.
            normalize_embeddings=True #normalizes the vectors so they're on a standardized scale.
        )

        return embeddings.tolist() #SentenceTransformer may return a NumPy array.Chroma/LangChain expects normal Python lists.

    def embed_query(self, text: str) -> list[float]: #This is for one user question.
        """
        Convert a user query into an embedding.
        """
        embedding = self.model.encode(
            text, #converts the user question into an embedding using the SentenceTransformer model.
            normalize_embeddings=True
        )

        return embedding.tolist() #Again, convert the NumPy result into a normal Python list


def create_vectorstore( #This function creates our actual Chroma database.
    documents: list[Document],
    embedding_model_name: str,
) -> Chroma: #means the function returns a Chroma vector store.
    """
    Create a persistent Chroma vector store
    from the provided document chunks.
    """

    PERSIST_DIRECTORY.mkdir( #This creates:data/vectorstore/ if it doesn't already exist
        parents=True, #Create parent directories if necessary.
        exist_ok=True #Don't give an error if the folder already exists.
    )

    embedding_function = LocalEmbeddingFunction(
        embedding_model_name
    )#This loads our model and creates the adapter.

    vectorstore = Chroma.from_documents(#This tells Chroma:Create a vector database from these documents
        documents=documents,#Here are our 1,568 chunks.
        embedding=embedding_function, #Use our local Sentence Transformer to convert those chunks into vectors
        collection_name=COLLECTION_NAME, #Store them in the rag_documents collection
        persist_directory=str(PERSIST_DIRECTORY), #Save the vector database to data/vectorstore.
    )

    return vectorstore
    