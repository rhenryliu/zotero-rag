"""Dynamic LanceDB schema for stored chunks.

``Vector(dim)`` is fixed at class-definition time, so the schema class is built
per active embedder dimension via :func:`make_chunk_model`.
"""

from __future__ import annotations

from lancedb.pydantic import LanceModel, Vector


def make_chunk_model(dim: int):
    """Build a LanceDB schema class for a given vector dimension.

    ``Vector(dim)`` is fixed at class-definition time, so the schema is built
    per active embedder dimension.

    Args:
        dim: Embedding dimension.

    Returns:
        A ``LanceModel`` subclass with a ``Vector(dim)`` field.
    """

    class Chunk(LanceModel):
        id: str
        doc_id: str
        title: str
        year: str
        page: int
        page_end: int
        pdf_path: str
        text: str
        vector: Vector(dim)  # type: ignore[valid-type]

    return Chunk
