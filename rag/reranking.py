"""Cross-encoder reranking (sentence-transformers, on MPS with a CPU fallback).

The shared primitive is :func:`_cross_encoder_scores`, which scores candidates
once per retrieval and degrades gracefully when reranking is disabled or the
model fails. :func:`rerank` is a thin, standalone convenience that is no longer
on the hot path (see its deprecation note).
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache

from .config import RERANK_DEVICE, RERANKER, RERANKERS, USE_RERANKER


@lru_cache(maxsize=1)
def _get_reranker():
    """Load the active cross-encoder once, with a CPU fallback and smoke-test.

    Returns:
        A loaded ``sentence_transformers.CrossEncoder``.
    """
    import torch
    from sentence_transformers import CrossEncoder

    spec = RERANKERS[RERANKER]
    kwargs: dict = {"trust_remote_code": spec.trust_remote_code}
    if spec.sigmoid:
        kwargs["activation_fn"] = torch.nn.Sigmoid()
    # Smoke-test inside the try so a predict-time failure on MPS (e.g. an
    # unsupported op, the known risk for gte-reranker-modernbert-base) also
    # falls back to CPU; only a CPU failure propagates.
    try:
        model = CrossEncoder(spec.model, device=RERANK_DEVICE, **kwargs)
        model.predict([("warmup query", "warmup passage")])
    except Exception:
        model = CrossEncoder(spec.model, device="cpu", **kwargs)
        model.predict([("warmup query", "warmup passage")])
    return model


def _cross_encoder_scores(question: str, hits: list[dict]) -> list[float] | None:
    """Score candidates with the active cross-encoder, or ``None`` if unavailable.

    Shared by :func:`rerank` and ``select_diverse`` so the (potentially slow)
    cross-encoder runs once per retrieval. Honours ``USE_RERANKER`` and degrades
    gracefully: returns ``None`` when reranking is disabled, there are no hits, or
    the model fails to load or predict.

    Args:
        question: The query (already rewritten, in chat mode).
        hits: Candidate chunk dicts from vector search.

    Returns:
        One relevance score per hit (same order), or ``None`` if the cross-encoder
        is unavailable.
    """
    if not USE_RERANKER or not hits:
        return None
    try:
        model = _get_reranker()
        scores = model.predict([(question, h["text"]) for h in hits])
    except Exception as exc:
        print(
            f"WARNING: cross-encoder unavailable ({exc!r}); falling back to vector order.",
            file=sys.stderr,
        )
        return None
    return [float(s) for s in scores]


def rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-order candidates by cross-encoder relevance, with graceful fallback.

    .. deprecated::
        DEAD CODE -- this function is not called anywhere in the pipeline and is
        slated for removal. ``RAGPipeline.retrieve`` scores via
        :func:`_cross_encoder_scores` (the shared primitive) and inlines its own
        score/dedup/select sequence, so it does not call this. Retained only as a
        standalone convenience for one-off scripting; calling it emits a
        ``DeprecationWarning`` and runs a fresh cross-encoder pass (it re-scores
        ``hits`` internally).

    Args:
        question: The query (already rewritten, in chat mode).
        hits: Candidate chunk dicts from vector search.
        top_k: Number of chunks to keep.

    Returns:
        The ``top_k`` most relevant chunks. Falls back to vector order if the
        reranker is disabled or fails to load.
    """
    warnings.warn(
        "rerank() is deprecated and unused; it will be removed in a future "
        "version. Use RAGPipeline.retrieve (or _cross_encoder_scores) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    scores = _cross_encoder_scores(question, hits)
    if scores is None:
        return hits[:top_k]
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [hit for hit, _ in ranked[:top_k]]
