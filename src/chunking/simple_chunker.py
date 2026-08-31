from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_simple_chunker() -> RecursiveCharacterTextSplitter:
    """
    Create the baseline recursive character text splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],#["paragraph", "line", "sentence", "word", "character"]sepertor
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller chunks.
    """

    splitter = create_simple_chunker()

    chunks = splitter.split_documents(documents)

    return chunks