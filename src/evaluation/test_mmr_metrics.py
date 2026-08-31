from src.evaluation.mmr_retrieval_eval import (
    evaluate_mmr_retrieval,
)
from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
)


results = evaluate_mmr_retrieval(k=5)


print("=" * 70)
print("MMR RETRIEVAL METRICS")
print("=" * 70)


print(
    f"Recall@1: "
    f"{recall_at_k(results, 1):.2%}"
)

print(
    f"Recall@3: "
    f"{recall_at_k(results, 3):.2%}"
)

print(
    f"Recall@5: "
    f"{recall_at_k(results, 5):.2%}"
)

print(
    f"MRR:      "
    f"{mean_reciprocal_rank(results):.4f}"
)