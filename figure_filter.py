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
        text = extract_page_text(page, extra_regions=table_rects)
    where ``table_rects`` are the bounding boxes of tables already rendered to
    Markdown (so tabular text is dropped from the prose in favour of that
    rendering). Everything downstream (paragraph-aware chunking on blank-line
    breaks, the separate ``find_tables()`` path) is unchanged: reading order and
    paragraph spacing match ``get_text("text")`` because dict blocks are
    returned in the same reading order.

Requires PyMuPDF >= 1.24 for ``cluster_drawings()``; older versions degrade
gracefully to raster-image regions only.
"""

from __future__ import annotations

from collections.abc import Sequence

import fitz  # PyMuPDF

# Figure-detection tunables (module level so they can be overridden per call).
FIGURE_MIN_AREA_FRAC = 0.03  # figure-vs-equation area gate (tunable)
FIGURE_PAD = 14.0            # region inflation in points
FIGURE_GRANULARITY = "line"  # "line" | "block"


def _figure_regions(
    page: fitz.Page, blocks: list[dict], min_area_frac: float, pad: float
) -> list[fitz.Rect]:
    """Find large graphic regions on a page: raster images + vector clusters.

    Only regions at or above ``min_area_frac`` of the page area qualify, which
    excludes small vector clusters such as display equations.

    Args:
        page: A PyMuPDF page (used for the page area and vector-drawing clusters).
        blocks: The ``blocks`` list from a single ``page.get_text("dict")`` call,
            passed in so the page dict is parsed only once per page.
        min_area_frac: Minimum region area as a fraction of total page area.
        pad: Points to inflate each region by, to catch axis tick labels and
            legends sitting just outside the plotting box.

    Returns:
        Inflated bounding rectangles of the qualifying figure regions.
    """
    page_area = page.rect.width * page.rect.height
    rects = [fitz.Rect(b["bbox"]) for b in blocks if b["type"] == 1]
    try:
        rects += list(page.cluster_drawings())
    except Exception:
        pass  # older PyMuPDF: raster image blocks only
    regions = []
    for r in rects:
        if (r.width * r.height) / page_area >= min_area_frac:
            regions.append(r + (-pad, -pad, pad, pad))
    return regions


def extract_page_text(
    page: fitz.Page,
    *,
    extra_regions: Sequence[fitz.Rect] = (),
    granularity: str = FIGURE_GRANULARITY,
    min_area_frac: float = FIGURE_MIN_AREA_FRAC,
    pad: float = FIGURE_PAD,
) -> str:
    """Extract a page's text in reading order, minus text inside figures.

    The page dict is parsed exactly once and shared between figure-region
    detection and text collection. Detected figure regions (raster images plus
    area-gated vector-drawing clusters) are unioned with ``extra_regions`` -- the
    caller passes table bounding boxes here so tabular text is dropped from the
    prose in favour of the Markdown rendering produced separately.

    Equation blocks survive because equation vector clusters fall below
    ``min_area_frac`` and are never flagged as figures.

    Args:
        page: A PyMuPDF page.
        extra_regions: Additional regions (e.g. table bounding boxes) to exclude,
            unioned with the detected figure regions.
        granularity: ``"line"`` drops only individual lines whose centre lies in a
            region (protecting captions that share a block with in-plot
            annotations); ``"block"`` drops a whole block when its centre lies in
            a region.
        min_area_frac: Minimum figure area as a fraction of page area. Lower to
            catch smaller figures (risks clipping large equation arrays);
            raise to be more conservative.
        pad: Points to inflate figure regions by when testing containment.

    Returns:
        Figure-filtered page text, paragraphs separated by blank lines.

    Raises:
        ValueError: If ``granularity`` is not ``"line"`` or ``"block"``.
    """
    if granularity not in {"line", "block"}:
        raise ValueError(f"granularity must be 'line' or 'block', got {granularity!r}")
    blocks: list[dict] = page.get_text("dict")["blocks"]  # type: ignore[index]
    regions = _figure_regions(page, blocks, min_area_frac, pad) + list(extra_regions)
    paragraphs: list[str] = []
    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue
        if granularity == "block":
            rect = fitz.Rect(block["bbox"])
            centre = (rect.tl + rect.br) / 2
            if any(centre in r for r in regions):
                continue  # whole block lives inside a figure
            lines = ["".join(span["text"] for span in ln["spans"]) for ln in block.get("lines", [])]
        else:  # "line": drop only the lines whose centre lies in a region
            lines = []
            for ln in block.get("lines", []):
                rect = fitz.Rect(ln["bbox"])
                centre = (rect.tl + rect.br) / 2
                if any(centre in r for r in regions):
                    continue  # in-plot annotation sharing a block with a caption
                lines.append("".join(span["text"] for span in ln["spans"]))
        text = "\n".join(lines).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)