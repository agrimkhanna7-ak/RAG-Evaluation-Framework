from src.evaluation.dataset import load_evaluation_dataset
from src.retrieval.hybrid_retriever import hybrid_search


def evaluate_hybrid_retrieval(k: int = 5):
    """
    Evaluate hybrid retrieval using the expected source/page.
    """

    dataset = load_evaluation_dataset()

    results = []

    for item in dataset:

        query = item["question"]
        expected_source = item["source"]
        expected_page = item["page"]

        retrieved = hybrid_search(
            query=query,
            k=k,
        )

        retrieved_documents = [
            document
            for document, score in retrieved
        ]

        first_exact_page_rank = None
        first_correct_document_rank = None

        for rank, document in enumerate(
            retrieved_documents,
            start=1,
        ):

            source = document.metadata.get("source")
            page = document.metadata.get("page")

            # Check whether the exact expected source and page were retrieved
            if (
                source == expected_source
                and page == expected_page
                and first_exact_page_rank is None
            ):
                first_exact_page_rank = rank

            # Check whether the correct PDF was retrieved
            if (
                source == expected_source
                and first_correct_document_rank is None
            ):
                first_correct_document_rank = rank

        results.append(
            {
                "id": item["id"],
                "question": query,
                "expected_source": expected_source,
                "expected_page": expected_page,
                "first_exact_page_rank": first_exact_page_rank,
                "first_correct_document_rank": first_correct_document_rank,
            }
        )

    return results