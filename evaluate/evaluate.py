from typing import List
import json

# caculate recall ratio
def single_recall_ratio(retrieve_results: List[str], ground_truth: list[str]) -> float:
    if len(ground_truth) == 0:
        return 0.0
    return len(set(retrieve_results) & set(ground_truth)) / len(ground_truth)

def total_recall_ratio(retrieve_results: List[List[str]], ground_truths: List[List[str]]) -> float:
    total_recall_ratio = 0.0
    for retrieve_result, ground_truth in zip(retrieve_results, ground_truths):
        total_recall_ratio += single_recall_ratio(retrieve_result, ground_truth)
    return total_recall_ratio / len(retrieve_results)

def evaluate_recall_ratio(data, reranked=False) -> float:
    retrieve_results = data
    ground_truths = [item["evidence_text"] for item in retrieve_results]
    if reranked:
        retrieve_results = [item["rerank_results"] for item in retrieve_results]
    else:
        retrieve_results = [item["retrieve_text"] for item in retrieve_results]
    recall_ratio = total_recall_ratio(retrieve_results, ground_truths)
    return recall_ratio


