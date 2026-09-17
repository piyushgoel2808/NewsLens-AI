"""Cross-Encoder Neural Reranker for Two-Stage Retrieval Cascade."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def _detect_best_device() -> str:
    """Detect available compute accelerator (CUDA on Linux/Windows, or CPU for optimal MiniLM throughput).

    Note: On macOS, CPU is explicitly preferred over MPS for small CrossEncoder models
    (ms-marco-MiniLM-L-6-v2) because CPU executes in ~80ms without the 10-15s Metal shader
    compilation lag and memory synchronization stalls inherent to MPS.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class HeuristicReranker:
    """Deterministic lexical and keyword-overlap reranker used as a fallback."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using token overlap and lexical proximity."""
        if not candidates:
            return []
        if len(candidates) == 1:
            cand = dict(candidates[0])
            cand.setdefault("rerank_score", cand.get("rrf_score", 0.0))
            return [cand]

        query_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower()))
        scored: list[tuple[float, dict[str, Any]]] = []

        for cand in candidates:
            text = f"{cand.get('headline', '')} {cand.get('snippet', '')} {cand.get('full_text', '')}".lower()
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text))
            overlap = len(query_tokens.intersection(doc_tokens))
            overlap_score = overlap / max(1, len(query_tokens))

            base_rrf = cand.get("rrf_score", 0.0)
            prominence = cand.get("prominence_score", 0.0)

            # Blended heuristic score
            final_score = (overlap_score * 0.6) + (base_rrf * 0.3) + (min(1.0, prominence) * 0.1)
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = round(final_score, 4)
            scored.append((final_score, cand_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class CrossEncoderReranker:
    """Production Cross-Encoder Neural Reranker utilizing sentence-transformers or fast fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or _detect_best_device()
        self._model: Any = None
        self._fallback = HeuristicReranker()
        self._load_lock = asyncio.Lock()

    def _load_model_sync(self) -> Any:
        """Load cross-encoder model synchronously."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading Cross-Encoder model on device",
                extra={"model": self.model_name, "device": self.device},
            )
            return CrossEncoder(self.model_name, device=self.device)
        except Exception as ex:
            logger.warning(
                "Failed to load sentence_transformers CrossEncoder, using heuristic fallback",
                extra={"error": str(ex)},
            )
            return None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronously compute raw cross-encoder relevance scores for query-document pairs."""
        if not pairs:
            return []
        model = self._model or self._load_model_sync()
        if model is not None:
            try:
                raw_scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
                return [float(s) for s in raw_scores]
            except Exception as ex:
                logger.warning("CrossEncoder predict failed, falling back to heuristic", extra={"error": str(ex)})
        # Heuristic fallback score calculation
        results: list[float] = []
        for q, doc in pairs:
            q_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q.lower()))
            d_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", doc.lower()))
            overlap = len(q_tokens.intersection(d_tokens))
            results.append(overlap / max(1, len(q_tokens)))
        return results

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank top candidates by computing cross-attention interaction scores."""
        if not candidates:
            return []
        if len(candidates) == 1:
            cand = dict(candidates[0])
            cand.setdefault("rerank_score", cand.get("rrf_score", 0.0))
            return [cand]

        model = await self._get_model()
        if model is None:
            return self._fallback.rerank(query, candidates, top_k=top_k)

        # Build query-document pairs
        pairs: list[tuple[str, str]] = []
        for cand in candidates:
            # Construct rich context representation
            headline = cand.get("headline", "")
            snippet = cand.get("snippet", "")
            full_text = cand.get("full_text", "")
            text_context = (
                f"{headline}\n{snippet}" if snippet else (headline + "\n" + full_text[:500])
            )
            pairs.append((query, text_context.strip()))

        try:
            scores = await asyncio.to_thread(
                model.predict,
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
            scored: list[tuple[float, dict[str, Any]]] = []
            for score, cand in zip(scores, candidates, strict=False):
                cand_copy = dict(cand)
                score_val = float(score)
                cand_copy["rerank_score"] = round(score_val, 4)
                scored.append((score_val, cand_copy))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:top_k]]
        except Exception as ex:
            logger.warning(
                "CrossEncoder inference failed, falling back to heuristic",
                extra={"error": str(ex)},
            )
            return self._fallback.rerank(query, candidates, top_k=top_k)
