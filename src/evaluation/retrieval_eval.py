from src.evaluation.dataset import load_evaluation_dataset
from src.retrieval.vector_retriever import retrieve_documents


def evaluate_retrieval(k: int = 5):
    """
    Evaluate whether the expected source/page
    appears in the top-k retrieved documents.
    """

    dataset = load_evaluation_dataset()

    results = []#store the evaluation result for every question here.

    for item in dataset:#take each question from our evaluation dataset one at a time.

        query = item["question"] #Get the question
        expected_source = item["source"] #Get expected source
        expected_page = item["page"] #Get expected page

        retrieved = retrieve_documents(#Run the retriever
            query=query,
            k=k,
        )# output of this would look like [(Document(page=25), 0.44),(Document(page=25), 0.45),]etc

        retrieved_documents = [
            document
            for document, score in retrieved
        ]#Extract only the documents

        found = False#We haven't found the expected page yet.If we find it, we'll change it to True.

        first_relevant_rank = None
        first_source_rank = None

        for rank, (document, score) in enumerate(retrieved, start=1):

            source = document.metadata.get("source")#Get source from the retrieved document
            page = document.metadata.get("page") #Get page from the retrieved document

            # Check whether the exact expected page was retrieved
            if (
                source == expected_source
                and page == expected_page
            ):
                found = True#If both source and page match

                if first_relevant_rank is None:
                    first_relevant_rank = rank

            # Check whether the correct PDF was retrieved
            if (
                source == expected_source
                and first_source_rank is None
            ):
                first_source_rank = rank

        results.append(
            {
                "id": item["id"],
                "question": query,
                "expected_source": expected_source,
                "expected_page": expected_page,
                "retrieved": found,
                "first_relevant_rank": first_relevant_rank,
                "first_source_rank": first_source_rank,
            }
        )

    return results