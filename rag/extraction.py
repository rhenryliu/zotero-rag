"""PDF extraction: figure-aware per-page text + tables, and page-image rendering.

``extract_pages`` calls ``find_tables()`` ONCE per page and feeds the result to
BOTH the Markdown chunk and the prose-exclusion regions (via
:func:`rag.figure_filter.extract_page_text`); a table whose Markdown is empty is
NOT excluded, so no content is dropped without a Markdown replacement. The
page-image helpers serve MULTIMODAL generation only -- retrieval stays text-only.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .config import IMAGE_DPI, MAX_IMAGES
from .figure_filter import extract_page_text


def extract_pages(pdf_path: Path):
    """Yield per-page content from a PDF.

    Tables are detected once per page via ``find_tables()`` and the result drives
    both outputs: each non-empty table is rendered to Markdown AND its bounding
    box is excluded from the prose (via :func:`rag.figure_filter.extract_page_text`),
    so a table's cells are not also re-extracted as garbled prose. A table whose
    Markdown is empty is left untouched (its region is NOT excluded) so no
    content is dropped without a replacement.

    Args:
        pdf_path: Path to the PDF.

    Yields:
        ``(page_number, text, tables_markdown)`` with 1-based page numbers and a
        list of Markdown tables found on the page. Yields nothing if unreadable.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return
    try:
        for pno, page in enumerate(doc, start=1):  # type: ignore[arg-type]
            tables: list[str] = []
            table_rects: list[fitz.Rect] = []
            try:
                for table in page.find_tables().tables:
                    markdown = table.to_markdown()
                    if markdown and markdown.strip():
                        tables.append(markdown.strip())
                        table_rects.append(fitz.Rect(table.bbox))
            except Exception:
                pass
            try:
                text = extract_page_text(page, extra_regions=table_rects)
            except Exception:
                text = ""  # skip this page's text on failure rather than abort
            if text.strip() or tables:
                yield pno, text, tables
    finally:
        doc.close()  # never leak the file handle on a mid-iteration error


def render_page_png(pdf_path: str, page: int, dpi: int = IMAGE_DPI) -> bytes:
    """Render a single PDF page to PNG bytes.

    Args:
        pdf_path: Path to the source PDF.
        page: 1-based page number.
        dpi: Render resolution.

    Returns:
        PNG-encoded image bytes.
    """
    doc = fitz.open(pdf_path)
    try:
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def collect_page_images(hits: list[dict], max_images: int = MAX_IMAGES) -> list[bytes]:
    """Render the unique pages of the top hits to PNG bytes, capped.

    A hit may span a page range ``[page, page_end]`` (cross-page chunks), so every
    page in the range is rendered, in order, de-duplicated across hits and capped
    at ``max_images``.

    Args:
        hits: Retrieved chunk dicts (each has ``pdf_path``, ``page`` and
            ``page_end``).
        max_images: Maximum number of page images.

    Returns:
        PNG byte blobs, one per unique page, up to ``max_images``.
    """
    seen: list[tuple[str, int]] = []
    for hit in hits:
        for page in range(int(hit["page"]), int(hit["page_end"]) + 1):
            key = (hit["pdf_path"], page)
            if key not in seen:
                seen.append(key)
    seen = seen[:max_images]
    images: list[bytes] = []
    for pdf_path, page in seen:
        try:
            images.append(render_page_png(pdf_path, page))
        except Exception:
            continue
    return images
