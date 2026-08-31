from rank_bm25 import BM25Okapi

from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.simple_chunker import chunk_documents
from src.retrieval.vector_retriever import retrieve_documents


# ============================================================
# LOAD AND CHUNK DOCUMENTS ONCE
# ============================================================

# Load all PDF pages from the documents directory
documents = load_all_pdfs()

# Split the page-level documents into the same chunks
# used by our RAG system
CHUNKS = chunk_documents(documents)


def tokenize(text: str) -> list[str]:
    """
    Convert text into tokens for BM25 keyword search.
    """

    return text.lower().split()


# Create tokenized versions of all chunks
# BM25 requires tokenized documents
TOKENIZED_CHUNKS = [
    tokenize(document.page_content)
    for document in CHUNKS
]


# Build the BM25 index once
# This avoids rebuilding the BM25 index for every question
BM25 = BM25Okapi(
    TOKENIZED_CHUNKS
)


def document_key(document) -> str:
    """
    Create a stable identifier for a document chunk.
    """

    source = document.metadata.get(
        "source",
        "",
    )

    page = document.metadata.get(
        "page",
        "",
    )

    # Use source + page + content so that the same
    # chunk can be matched between semantic and BM25 results
    return (
        f"{source}|{page}|"
        f"{document.page_content}"
    )


def hybrid_search(
    query: str,
    k: int = 5,
):
    """
    Retrieve documents using both semantic and keyword search.
    """

    # Number of candidates retrieved from each search method
    # before combining their rankings.
    candidate_k = 20


    # ========================================================
    # 1. SEMANTIC SEARCH
    # ========================================================

    # Retrieve the top 20 chunks using semantic similarity.
    # Semantic search understands the meaning of the query.
    semantic_results = retrieve_documents(
        query=query,
        k=candidate_k,
    )

    # Extract only the documents from the
    # (document, score) results.
    semantic_ranked = [
        document
        for document, score in semantic_results
    ]


    # ========================================================
    # 2. KEYWORD SEARCH USING BM25
    # ========================================================

    # Convert the query into tokens for BM25.
    query_tokens = tokenize(query)

    # Calculate BM25 scores for the query
    # against all document chunks.
    bm25_scores = BM25.get_scores(
        query_tokens
    )

    # Combine each document with its BM25 score
    # and sort from highest score to lowest score.
    keyword_ranked = sorted(
        zip(CHUNKS, bm25_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Keep only the top 20 BM25 candidates.
    keyword_ranked = [
        document
        for document, score in keyword_ranked[
            :candidate_k
        ]
    ]


    # ========================================================
    # 3. WEIGHTED RECIPROCAL RANK FUSION
    # ========================================================

    # Dictionary used to store the combined RRF score
    # for every candidate document.
    rrf_scores = {}

    # RRF constant.
    # A larger value reduces the difference between
    # higher and lower ranks.
    rrf_k = 60


    # --------------------------------------------------------
    # Add semantic ranking
    # --------------------------------------------------------

    # Semantic search gets 70% of the hybrid influence
    # because semantic retrieval is our stronger baseline.
    for rank, document in enumerate(
        semantic_ranked,
        start=1,
    ):

        key = document_key(document)

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 0.7 / (rrf_k + rank)
        )


    # --------------------------------------------------------
    # Add BM25 ranking
    # --------------------------------------------------------

    # BM25 gets 30% of the hybrid influence.
    # This allows exact keyword matching to contribute
    # without overpowering semantic retrieval.
    for rank, document in enumerate(
        keyword_ranked,
        start=1,
    ):

        key = document_key(document)

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 0.3 / (rrf_k + rank)
        )


    # ========================================================
    # 4. CREATE FINAL HYBRID RANKING
    # ========================================================

    # Create a dictionary that maps the stable document key
    # back to the actual Document object.
    all_documents = {
        document_key(document): document
        for document in CHUNKS
    }

    # Sort documents according to their combined
    # weighted RRF score.
    hybrid_ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


    # ========================================================
    # 5. RETURN FINAL TOP-K DOCUMENTS
    # ========================================================

    results = []

    # Take only the final top-k hybrid results.
    for key, score in hybrid_ranked[:k]:

        document = all_documents[key]

        results.append(
            (
                document,
                score,
            )
        )

    return results