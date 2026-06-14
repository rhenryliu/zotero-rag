"""Diagnose multi-column reading-order quality on a single PDF.

Extracts the same pages two ways and prints them for side-by-side eyeballing:

    [A] ``page.get_text("text")``        -- exactly what the current RAG
                                            pipeline uses; the baseline to judge.
    [B] ``pymupdf4llm.to_markdown(...)`` -- layout-aware extraction with
                                            column / reading-order reconstruction.

Read each page's two outputs and check whether sentences flow naturally. If the
baseline interleaves columns -- a left-column sentence followed by an unrelated
right-column one -- reading order is broken and pymupdf4llm is worth adopting.
If the baseline already reads cleanly, plain extraction is fine and there is
nothing to gain.

Usage:
    Edit PDF_PATH below, then run:
        python pdf_extract_test.py
    Or pass a path (and optional page spec) on the command line:
        python pdf_extract_test.py /path/to/paper.pdf
        python pdf_extract_test.py /path/to/paper.pdf 1-3

Requires: pymupdf. Optional: pymupdf4llm (pip install pymupdf4llm); without it
the script prints the baseline only.
"""

from __future__ import annotations

import os
import sys

import fitz  # PyMuPDF


class _SuppressFD:
    """Silence C-level stdout/stderr by redirecting fds 1 and 2 to os.devnull.

    PyMuPDF / MuPDF print parser and OCR messages from the underlying C library,
    below Python's ``sys.stdout``, so a Python-level redirect does not catch
    them. Redirecting the OS file descriptors does. File descriptors are always
    restored on exit; exceptions raised inside the block still propagate.
    """

    def __enter__(self) -> "_SuppressFD":
        self._null = os.open(os.devnull, os.O_WRONLY)
        self._saved = (os.dup(1), os.dup(2))
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc) -> None:
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for fd in (self._null, *self._saved):
            os.close(fd)

# --- edit these ------------------------------------------------------------
PDF_PATH = "/Users/hliu/projects/papers/Battaglia2012.pdf"  # paste your PDF path here
PAGES = "3-4"                         # pages to inspect (1-based), e.g. "1-3" or "2"
MAX_CHARS_PER_PAGE = 2000             # cap per method per page; raise to see more
# ---------------------------------------------------------------------------


def parse_pages(spec: str, page_count: int) -> list[int]:
    """Turn a page spec like ``"1-3"`` or ``"2"`` into 0-based indices.

    Args:
        spec: Page spec, 1-based and inclusive (e.g. ``"1-3"`` or ``"2"``).
        page_count: Total number of pages in the document.

    Returns:
        Sorted list of valid 0-based page indices.
    """
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        start, end = int(lo), int(hi)
    else:
        start = end = int(spec)
    return [i for i in range(start - 1, end) if 0 <= i < page_count]


def clip(text: str, limit: int) -> str:
    """Trim text to ``limit`` characters with a visible marker when truncated.

    Args:
        text: Text to trim.
        limit: Maximum characters to keep.

    Returns:
        The trimmed text, annotated if truncation occurred.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... [truncated at {limit} chars]"


def plain_text_pages(pdf_path: str, indices: list[int]) -> dict[int, str]:
    """Extract pages with ``get_text("text")`` -- the current-pipeline method.

    Args:
        pdf_path: Path to the PDF.
        indices: 0-based page indices to extract.

    Returns:
        Mapping of 0-based page index to extracted text.
    """
    out: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for i in indices:
            out[i] = doc[i].get_text("text")
    finally:
        doc.close()
    return out


def pymupdf4llm_pages(pdf_path: str, indices: list[int]) -> dict[int, str] | None:
    """Extract pages with pymupdf4llm's layout-aware Markdown, if installed.

    The library emits parser / OCR chatter from the C layer; it is silenced via
    file-descriptor redirection so the report stays clean. Each chunk's
    ``metadata["page_number"]`` is 1-based absolute (verified: requesting pages
    ``[0, 2]`` returns page numbers ``1`` and ``3``), so ``page_number - 1``
    maps back to this script's 0-based indexing regardless of chunk ordering.

    Args:
        pdf_path: Path to the PDF.
        indices: 0-based page indices to extract.

    Returns:
        Mapping of 0-based page index to Markdown text, or ``None`` if
        pymupdf4llm is not installed.
    """
    try:
        import pymupdf4llm
    except ImportError:
        return None
    with _SuppressFD():
        chunks = pymupdf4llm.to_markdown(
            pdf_path,
            pages=sorted(indices),  # parse only the pages we need
            page_chunks=True,
            show_progress=False,
        )
    return {c["metadata"]["page_number"] - 1: c.get("text", "") for c in chunks}


def main() -> None:
    """Run the comparison and print a report for the requested pages."""
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    pages_spec = sys.argv[2] if len(sys.argv) > 2 else PAGES

    if not os.path.exists(pdf_path):
        sys.exit(
            f"File not found: {pdf_path!r}\n"
            "Edit PDF_PATH at the top, or pass a path as the first argument."
        )

    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    indices = parse_pages(pages_spec, page_count)
    if not indices:
        sys.exit(f"No valid pages in spec {pages_spec!r} for a {page_count}-page document.")

    print(f"PDF: {pdf_path}")
    print(
        f"Pages in document: {page_count} | inspecting (1-based): "
        f"{', '.join(str(i + 1) for i in indices)}"
    )
    print("=" * 78)

    plain = plain_text_pages(pdf_path, indices)
    layout = pymupdf4llm_pages(pdf_path, indices)
    if layout is None:
        print("NOTE: pymupdf4llm is not installed -- showing baseline [A] only.")
        print("      Install with:  pip install pymupdf4llm")
        print("=" * 78)

    for i in indices:
        print(f"\n########## PAGE {i + 1} ##########")
        print("\n----- [A] get_text('text')  (current pipeline) -----")
        print(clip(plain.get(i, ""), MAX_CHARS_PER_PAGE) or "[no extractable text on this page]")
        if layout is not None:
            print("\n----- [B] pymupdf4llm.to_markdown  (layout-aware) -----")
            print(clip(layout.get(i, ""), MAX_CHARS_PER_PAGE) or "[no extractable text on this page]")

    print("\n" + "=" * 78)
    print("What to look for: in [A], do sentences flow, or does a left-column")
    print("sentence jump to an unrelated right-column one (interleaving)? If [A]")
    print("reads cleanly, plain extraction is fine. If [A] scrambles but [B]")
    print("reads in order, that is the case for adopting pymupdf4llm.")


if __name__ == "__main__":
    main()