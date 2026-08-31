def recall_at_k(results, k: int) -> float:
    """
    Calculate Recall@K.
    """

    if not results:#this checks whether the results list is empty.
        return 0.0

    relevant = 0 #This is our counter.How many questions had a relevant document within the top K?

    for result in results: #Now we loop through every evaluation result.

        rank = result["first_relevant_rank"] #We're extracting the rank we calculated earlier in retrieval_eval.py.

        if rank is not None and rank <= k: #Did we find a relevant document at all and Was the relevant document found within the first K results?
            relevant += 1 #If the question passes the previous test, we increase our counter

    return relevant / len(results) #This calculates the final Recall@K.


def mean_reciprocal_rank(results) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    """

    if not results:
        return 0.0

    reciprocal_ranks = []

    for result in results:

        rank = result["first_relevant_rank"]

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)