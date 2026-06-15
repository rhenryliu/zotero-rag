"""Text chunking and the embedding-size guard chain.

Prose is packed across page breaks by :func:`chunk_document`; tables are kept
per-page and never routed through it. A chunk larger than the embedding runner's
batch crashes it (EOF), so :func:`_hard_split` and :func:`split_oversized_records`
enforce ``MAX_EMBED_CHARS`` as the final net.
"""

from __future__ import annotations

import re
import warnings

from .config import CHUNK_CHARS, CHUNK_OVERLAP, MAX_EMBED_CHARS, MIN_CHUNK_CHARS


def _hard_split(text: str, limit: int) -> list[str]:
    """Split text into pieces no longer than ``limit`` characters.

    Prefers sentence boundaries, falling back to a hard character cut for any
    single sentence that alone exceeds the limit, so no piece can blow past the
    embedding runner's batch limit.

    Args:
        text: Text to split.
        limit: Maximum characters per piece.

    Returns:
        Non-empty pieces, each at most ``limit`` characters.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > limit:  # a single over-long sentence: hard-cut it
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            pieces.append(sentence[:limit].strip())
            sentence = sentence[limit:]
        if buffer and len(buffer) + len(sentence) + 1 > limit:
            pieces.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer.strip():
        pieces.append(buffer.strip())
    return [p for p in pieces if p]


def chunk_page(text: str) -> list[str]:
    """Split a page's text into overlapping, paragraph-aware chunks.

    .. deprecated::
        DEAD CODE -- this function is not called anywhere in the pipeline and is
        slated for removal. Ingestion chunks across page breaks via
        :func:`chunk_document`, not page-by-page; this single-page variant is
        retained only for one-off scripting and emits a ``DeprecationWarning``
        when called.

    Paragraphs longer than ``CHUNK_CHARS`` are hard-split first, so a single
    giant block -- common in dense, multi-column, equation-heavy PDFs where
    PyMuPDF finds no paragraph breaks -- can't become one oversized chunk that
    crashes the embedding runner. Pages whose paragraphs are all within the cap
    chunk identically to before.

    Args:
        text: Raw page text.

    Returns:
        Chunk strings, each at least ``MIN_CHUNK_CHARS`` long.
    """
    warnings.warn(
        "chunk_page() is deprecated and unused; it will be removed in a future "
        "version. Ingestion uses chunk_document() (cross-page packing) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    bounded: list[str] = []
    for para in paragraphs:
        bounded.extend(_hard_split(para, CHUNK_CHARS) if len(para) > CHUNK_CHARS else [para])
    paragraphs = bounded

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) + 1 > CHUNK_CHARS:
            chunks.append(buffer.strip())
            buffer = buffer[-CHUNK_OVERLAP:] + " " + para
        else:
            buffer = f"{buffer} {para}".strip()
    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_document(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Chunk a whole document so one idea can straddle a page break.

    Each page's text is split on blank lines into paragraphs, each tagged with
    its source page, and the tagged paragraphs are concatenated into a single
    document-order stream. Packing then runs over that stream, so a paragraph
    continued across a page boundary lands in the same chunk as its
    continuation instead of being cut at the page edge. Oversized paragraphs are
    hard-split first via :func:`_hard_split`, exactly as in :func:`chunk_page`,
    and packing uses the same budget/overlap rules (``CHUNK_CHARS``,
    ``CHUNK_OVERLAP``, ``MIN_CHUNK_CHARS``).

    Each chunk is tagged with the ``(min_page, max_page)`` of the paragraphs it
    contains. The overlap tail carried into a new chunk is attributed to the new
    paragraph's page; this minor mis-attribution of the carried tail is
    acceptable.

    Args:
        pages: ``(page_number, page_text)`` pairs in document order.

    Returns:
        ``(start_page, end_page, text)`` chunks, each at least ``MIN_CHUNK_CHARS``
        characters long.
    """
    # Document-order paragraph stream, each paragraph tagged with its page.
    tagged: list[tuple[int, str]] = []
    for page_no, text in pages:
        for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
            pieces = _hard_split(para, CHUNK_CHARS) if len(para) > CHUNK_CHARS else [para]
            tagged.extend((page_no, piece) for piece in pieces)

    chunks: list[tuple[int, int, str]] = []
    buffer = ""
    buffer_pages: set[int] = set()
    for page_no, para in tagged:
        if buffer and len(buffer) + len(para) + 1 > CHUNK_CHARS:
            chunks.append((min(buffer_pages), max(buffer_pages), buffer.strip()))
            buffer = buffer[-CHUNK_OVERLAP:] + " " + para
            buffer_pages = {page_no}  # carried tail attributed to the new para's page
        else:
            buffer = f"{buffer} {para}".strip()
            buffer_pages.add(page_no)
    if buffer.strip():
        chunks.append((min(buffer_pages), max(buffer_pages), buffer.strip()))
    return [(s, e, c) for s, e, c in chunks if len(c) >= MIN_CHUNK_CHARS]


def split_oversized_records(records: list[dict]) -> list[dict]:
    """Hard-split any record whose text exceeds ``MAX_EMBED_CHARS``.

    Final safety net before embedding: after chunk_page and table handling this
    should rarely fire, but it guarantees no single input can crash the runner.
    Split sub-records get a deterministic ``#n`` id suffix so re-ingest stays
    idempotent.

    Args:
        records: Chunk record dicts (each with ``id`` and ``text``).

    Returns:
        Records with any oversized ones replaced by suffixed sub-records.
    """
    out: list[dict] = []
    for record in records:
        if len(record["text"]) <= MAX_EMBED_CHARS:
            out.append(record)
            continue
        for sidx, piece in enumerate(_hard_split(record["text"], CHUNK_CHARS)):
            sub = dict(record)
            sub["id"] = f"{record['id']}#{sidx}"
            sub["text"] = piece
            out.append(sub)
    return out
