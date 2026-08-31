from src.evaluation.retrieval_eval import evaluate_retrieval
from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
)


results = evaluate_retrieval(k=5)


recall_1 = recall_at_k(results, k=1)
recall_3 = recall_at_k(results, k=3)
recall_5 = recall_at_k(results, k=5)

mrr = mean_reciprocal_rank(results)


print("\n" + "=" * 70)
print("RETRIEVAL METRICS")
print("=" * 70)

print(f"Recall@1: {recall_1:.2%}")
print(f"Recall@3: {recall_3:.2%}")
print(f"Recall@5: {recall_5:.2%}")
print(f"MRR:      {mrr:.4f}")