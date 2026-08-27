"""Quantitative Information Retrieval (IR) and Generation Evaluation Metrics.

Includes mathematical metrics for:
- Recall@K, Precision@K, Mean Reciprocal Rank (MRR), NDCG@K
- Lexical and LLM-assisted Faithfulness
- Citation Precision and Recall
- Multi-Newspaper Coverage F1 Score
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any


def compute_recall_at_k(
    retrieved_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
    k: int,
) -> float:
    """Calculate Recall@K: Proportion of relevant items retrieved in top-K."""
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k_retrieved = {str(item) for item in retrieved_ids[:k]}
    gt_set = {str(item) for item in ground_truth_ids}
    hits = len(top_k_retrieved.intersection(gt_set))
    return hits / len(gt_set)


def compute_precision_at_k(
    retrieved_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
    k: int,
) -> float:
    """Calculate Precision@K: Proportion of top-K retrieved items that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    if not ground_truth_ids:
        return 0.0

    actual_k = min(k, len(retrieved_ids))
    top_k_retrieved = {str(item) for item in retrieved_ids[:actual_k]}
    gt_set = {str(item) for item in ground_truth_ids}
    hits = len(top_k_retrieved.intersection(gt_set))
    return hits / actual_k


def compute_mrr(
    retrieved_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant item."""
    if not retrieved_ids or not ground_truth_ids:
        return 0.0

    gt_set = {str(item) for item in ground_truth_ids}
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if str(item_id) in gt_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if k <= 0 or not retrieved_ids or not ground_truth_ids:
        return 0.0

    gt_set = {str(item) for item in ground_truth_ids}
    top_k = [str(item) for item in retrieved_ids[:k]]

    # Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item_id in enumerate(top_k):
        rel = 1.0 if item_id in gt_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)

    # Ideal Discounted Cumulative Gain (IDCG)
    ideal_hits = min(k, len(gt_set))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def compute_citation_precision(
    cited_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
) -> float:
    """Calculate precision of generated citation references against ground-truth sources."""
    if not cited_ids:
        return 1.0 if not ground_truth_ids else 0.0
    if not ground_truth_ids:
        return 0.0

    cited_set = {str(item) for item in cited_ids}
    gt_set = {str(item) for item in ground_truth_ids}
    valid_citations = len(cited_set.intersection(gt_set))
    return valid_citations / len(cited_set)


def compute_citation_recall(
    cited_ids: Sequence[Any],
    ground_truth_ids: Sequence[Any],
) -> float:
    """Calculate recall of generated citation references against ground-truth sources."""
    if not ground_truth_ids:
        return 1.0
    if not cited_ids:
        return 0.0

    cited_set = {str(item) for item in cited_ids}
    gt_set = {str(item) for item in ground_truth_ids}
    hits = len(cited_set.intersection(gt_set))
    return hits / len(gt_set)


def compute_faithfulness(
    generated_answer: str,
    context_chunks: Sequence[str],
) -> float:
    """Calculate lexical and factual faithfulness score of answer given context."""
    if not generated_answer or not generated_answer.strip():
        return 0.0
    if not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    answer_sentences = [
        s.strip() for s in re.split(r"[.!?\n]+", generated_answer) if len(s.strip()) > 10
    ]

    if not answer_sentences:
        return 1.0

    supported_count = 0
    for sent in answer_sentences:
        words = [
            w
            for w in re.findall(r"\b[a-zA-Z0-9_-]{4,}\b", sent.lower())
            if w not in {"that", "with", "this", "from", "were", "have", "been", "their", "which"}
        ]
        if not words:
            supported_count += 1
            continue

        matching_words = sum(1 for w in words if w in combined_context)
        overlap_ratio = matching_words / len(words)
        if overlap_ratio >= 0.50:
            supported_count += 1

    return supported_count / len(answer_sentences)


async def compute_faithfulness_llm_judge(
    generated_answer: str,
    context_chunks: Sequence[str],
    llm_provider: Any = None,
) -> float:
    """Evaluate factual grounding and hallucinations using an LLM-as-a-Judge."""
    if not generated_answer or not generated_answer.strip():
        return 0.0
    if not context_chunks:
        return 0.0

    # If no LLM provider supplied, fallback to lexical faithfulness
    if llm_provider is None:
        return compute_faithfulness(generated_answer, context_chunks)

    from app.providers.base import Message

    prompt = f"""You are an authoritative verification judge for a Newspaper RAG system.
Evaluate whether EVERY claim in the Answer is strictly supported by the Context.

CONTEXT:
\"\"\"
{" ".join(context_chunks[:5])}
\"\"\"

ANSWER:
\"\"\"
{generated_answer}
\"\"\"

Provide a single float score between 0.0 (completely hallucinated/unsupported) to 1.0 (fully grounded in context).
Respond with ONLY the numeric score (e.g. 0.95)."""

    try:
        res = await llm_provider.complete([Message(role="user", content=prompt)], max_tokens=10)
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", res.text)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
    except Exception:
        pass

    return compute_faithfulness(generated_answer, context_chunks)


def compute_coverage_f1(
    predicted_covered: set[str],
    ground_truth_covered: set[str],
    predicted_omitted: set[str],
    ground_truth_omitted: set[str],
) -> float:
    """Calculate macro-averaged F1 score for newspaper coverage and gap classification."""
    # Coverage classification precision/recall
    covered_tp = len(predicted_covered.intersection(ground_truth_covered))
    covered_fp = len(predicted_covered - ground_truth_covered)
    covered_fn = len(ground_truth_covered - predicted_covered)

    covered_prec = (
        covered_tp / (covered_tp + covered_fp) if (covered_tp + covered_fp) > 0 else 1.0
    )
    covered_rec = (
        covered_tp / (covered_tp + covered_fn) if (covered_tp + covered_fn) > 0 else 1.0
    )
    covered_f1 = (
        (2 * covered_prec * covered_rec / (covered_prec + covered_rec))
        if (covered_prec + covered_rec) > 0
        else 0.0
    )

    # Omission classification precision/recall
    omitted_tp = len(predicted_omitted.intersection(ground_truth_omitted))
    omitted_fp = len(predicted_omitted - ground_truth_omitted)
    omitted_fn = len(ground_truth_omitted - predicted_omitted)

    omitted_prec = (
        omitted_tp / (omitted_tp + omitted_fp) if (omitted_tp + omitted_fp) > 0 else 1.0
    )
    omitted_rec = (
        omitted_tp / (omitted_tp + omitted_fn) if (omitted_tp + omitted_fn) > 0 else 1.0
    )
    omitted_f1 = (
        (2 * omitted_prec * omitted_rec / (omitted_prec + omitted_rec))
        if (omitted_prec + omitted_rec) > 0
        else 0.0
    )

    return (covered_f1 + omitted_f1) / 2.0
