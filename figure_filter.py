"""Figure-aware page text extraction for the Zotero RAG pipeline.

Drop-in replacement for ``page.get_text("text")`` in ``extract_pages``. It
strips text living inside large figures (axis labels, legends, in-plot
annotations) while keeping prose, captions, tables, and -- critically --
display equations.

The key insight: pymupdf4llm drops equations because it treats their vector
glyphs as pictures. We avoid that failure by gating on AREA. A real figure is a
large graphic region (raster image, or a vector-drawing cluster covering a
sizeable fraction of the page); a display equation is a small vector cluster
well below that threshold, so it is never classified as a figure and its text
survives untouched.

Integration:
    In ``extract_pages``, replace
        text = page.get_text("text")
    with
        text = extract_page_text(page)
    Everything downstream (paragraph-aware ``chunk_page`` on blank-line breaks,
    the separate ``find_tables()`` path) is unchanged: reading order and
    paragraph spacing match ``get_text("text")`` because dict blocks are
    returned in the same reading order.

Requires PyMuPDF >= 1.24 for ``cluster_drawings()``; older versions degrade
gracefully to raster-image regions only.
"""

from __future__ import annotations

import fitz  # PyMuPDF


def _figure_regions(page: fitz.Page, min_area_frac: float, pad: float) -> list[fitz.Rect]:
    """Find large graphic regions on a page: raster images + vector clusters.

    Only regions at or above ``min_area_frac`` of the page area qualify, which
    excludes small vector clusters such as display equations.

    Args:
        page: A PyMuPDF page.
        min_area_frac: Minimum region area as a fraction of total page area.
        pad: Points to inflate each region by, to catch axis tick labels and
            legends sitting just outside the plotting box.

    Returns:
        Inflated bounding rectangles of the qualifying figure regions.
    """
    page_area = page.rect.width * page.rect.height
    rects = [fitz.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"] if b["type"] == 1]
    try:
        rects += list(page.cluster_drawings())
    except Exception:
        pass  # older PyMuPDF: raster image blocks only
    regions = []
    for r in rects:
        if (r.width * r.height) / page_area >= min_area_frac:
            regions.append(r + (-pad, -pad, pad, pad))
    return regions


def extract_page_text(page: fitz.Page, min_area_frac: float = 0.03, pad: float = 14.0) -> str:
    """Extract a page's text in reading order, minus text inside figures.

    A text block is dropped only if its centre lies within a detected figure
    region. Equation blocks survive because equation vector clusters fall below
    ``min_area_frac`` and are never flagged as figures.

    Args:
        page: A PyMuPDF page.
        min_area_frac: Minimum figure area as a fraction of page area. Lower to
            catch smaller figures (risks clipping large equation arrays);
            raise to be more conservative.
        pad: Points to inflate figure regions by when testing containment.

    Returns:
        Figure-filtered page text, paragraphs separated by blank lines.
    """
    regions = _figure_regions(page, min_area_frac, pad)
    paragraphs: list[str] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:  # skip image blocks
            continue
        rect = fitz.Rect(block["bbox"])
        centre = (rect.tl + rect.br) / 2
        if any(centre in r for r in regions):
            continue  # text living inside a figure
        lines = ["".join(span["text"] for span in ln["spans"]) for ln in block.get("lines", [])]
        text = "\n".join(lines).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)