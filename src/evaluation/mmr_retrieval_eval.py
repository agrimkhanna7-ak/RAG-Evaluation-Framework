from src.evaluation.dataset import load_evaluation_dataset
from src.retrieval.mmr_retriever import retrieve_mmr_documents


def evaluate_mmr_retrieval(k: int = 5):
    """
    Evaluate MMR retrieval using the expected source and page.
    """

    dataset = load_evaluation_dataset()

    results = []

    for item in dataset:

        query = item["question"]

        expected_source = item["source"]
        expected_page = item["page"]

        retrieved_documents = retrieve_mmr_documents(
            query=query,
            k=k,
            fetch_k=20,
            lambda_mult=0.5,
        )

        # This will store the rank at which the exact
        # expected source + page is first retrieved.
        first_relevant_rank = None

        # This stores the first rank where the correct
        # PDF/document is retrieved, regardless of page.
        first_correct_document_rank = None

        for rank, document in enumerate(
            retrieved_documents,
            start=1,
        ):

            source = document.metadata.get("source")
            page = document.metadata.get("page")

            # Check for the exact expected PDF and page.
            if (
                source == expected_source
                and page == expected_page
                and first_relevant_rank is None
            ):
                first_relevant_rank = rank

            # Check whether the correct PDF was retrieved.
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

                # IMPORTANT:
                # metrics.py expects this exact key.
                "first_relevant_rank": first_relevant_rank,

                "first_correct_document_rank":
                    first_correct_document_rank,
            }
        )

    return results