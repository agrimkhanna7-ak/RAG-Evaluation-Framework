from src.evaluation.hybrid_retrieval_eval import (
    evaluate_hybrid_retrieval,
)
from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
)


# Run hybrid retrieval evaluation
results = evaluate_hybrid_retrieval(k=5)


# Use the exact expected page as the relevance criterion
metric_results = []

for result in results:

    metric_results.append(
        {
            "first_relevant_rank":
                result["first_exact_page_rank"]
        }
    )


print("=" * 70)
print("HYBRID RETRIEVAL METRICS")
print("=" * 70)


recall_1 = recall_at_k(
    metric_results,
    k=1,
)

recall_3 = recall_at_k(
    metric_results,
    k=3,
)

recall_5 = recall_at_k(
    metric_results,
    k=5,
)

mrr = mean_reciprocal_rank(
    metric_results
)


print(
    f"Recall@1: {recall_1 * 100:.2f}%"
)

print(
    f"Recall@3: {recall_3 * 100:.2f}%"
)

print(
    f"Recall@5: {recall_5 * 100:.2f}%"
)

print(
    f"MRR:      {mrr:.4f}"
)