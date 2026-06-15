"""Embedding via Ollama, plus the active embedder's identity/dimension.

Documents are embedded with no instruction; queries get an ``Instruct:`` prefix
only for instruction-aware (Qwen) embedders. Vectors are truncated to
:func:`effective_dim` and L2-normalised, so plain L2 search ranks identically to
cosine (no metric is set at search time).
"""

from __future__ import annotations

import numpy as np
import ollama

from .config import (
    EMBED_BATCH,
    EMBED_NUM_CTX,
    EMBEDDER,
    EMBEDDERS,
    INSTRUCTION_TASK,
    MRL_TRUNCATE_4B,
    EmbedderSpec,
)


# === Embedder identity / dimension ==========================================


def _embedder_spec() -> EmbedderSpec:
    """Return the spec for the active embedder."""
    return EMBEDDERS[EMBEDDER]


def effective_dim() -> int:
    """Return the stored vector dimension for the active embedder.

    Equals the model's native dim, except the qwen3-4b is truncated to 1024 when
    ``MRL_TRUNCATE_4B`` is set.

    Returns:
        The embedding dimension actually written to LanceDB.
    """
    if EMBEDDER == "qwen3-4b" and MRL_TRUNCATE_4B:
        return 1024
    return _embedder_spec().dim


def embedder_id() -> str:
    """Return the identity key for the active embedder, e.g. ``qwen3-0.6b-1024``."""
    return f"{EMBEDDER}-{effective_dim()}"


# === Embedding (Ollama) =====================================================


def _embed_raw(texts: list[str]) -> list[list[float]]:
    """Embed texts via the active Ollama embedding model (no pre/post-processing).

    Args:
        texts: Texts to embed.

    Returns:
        Raw embedding vectors, one per input.
    """
    model = _embedder_spec().ollama_model
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        resp = ollama.embed(model=model, input=batch, options={"num_ctx": EMBED_NUM_CTX})
        embs = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
        vectors.extend([list(e) for e in embs])
    return vectors


def _postprocess(vectors: list[list[float]]) -> list[list[float]]:
    """Truncate to the effective dimension and L2-normalise (float32).

    Normalising means the default L2 distance ranks identically to cosine, so no
    explicit metric is needed at search time.

    Args:
        vectors: Raw embedding vectors.

    Returns:
        Truncated, unit-normalised float32 vectors as plain lists.
    """
    dim = effective_dim()
    arr = np.asarray(vectors, dtype=np.float32)[:, :dim]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk/document texts (no instruction prefix).

    Args:
        texts: Document texts.

    Returns:
        Processed embedding vectors.
    """
    return _postprocess(_embed_raw(texts))


def embed_query(text: str) -> list[float]:
    """Embed a query, prepending the instruction for instruction-aware embedders.

    Args:
        text: The query string.

    Returns:
        A single processed embedding vector.
    """
    if _embedder_spec().instruct:
        text = f"Instruct: {INSTRUCTION_TASK}\nQuery: {text}"
    return _postprocess(_embed_raw([text]))[0]
