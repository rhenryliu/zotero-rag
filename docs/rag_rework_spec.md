# RAG pipeline rework: unified extraction, cross-page chunking, diverse selection

Repo files in scope: `zotero_rag.py` (main), `figure_filter.py` (exists, currently
UNUSED — wire it in and extend it). Match existing style: Google-style docstrings,
type hints, `from __future__ import annotations`. Keep edits localized; do not
refactor unrelated code. A schema change is involved → a full `rebuild` is required
after; do NOT migrate existing tables, just note it.

## New config constants

In `figure_filter.py`, module level:
  FIGURE_MIN_AREA_FRAC = 0.03      # figure-vs-equation area gate (tunable)
  FIGURE_PAD           = 14.0      # region inflation in points
  FIGURE_GRANULARITY   = "line"    # "line" | "block"

In `zotero_rag.py` config section (near the retrieval knobs):
  SELECT_DIVERSE = True
  MMR_LAMBDA     = 0.7             # relevance vs within-doc redundancy
  PER_DOC_CAP    = 3              # soft max chunks per paper

`MULTIMODAL` stays False. GEN_MODEL comment unchanged.

## Phase 1 — figure_filter.py: line-level + table-region exclusion + one dict parse

Rework `extract_page_text` to this contract:

    extract_page_text(
        page, *, extra_regions=(), granularity=FIGURE_GRANULARITY,
        min_area_frac=FIGURE_MIN_AREA_FRAC, pad=FIGURE_PAD,
    ) -> str

Requirements:
- Call `page.get_text("dict")` ONCE; pass its blocks into `_figure_regions`
  (change `_figure_regions` to accept precomputed blocks instead of calling
  get_text itself). Today it parses the dict twice per page — fix that.
- Build figure regions as today (raster image blocks + cluster_drawings, area-
  gated, inflated by pad), then UNION with `extra_regions` (caller passes table
  bboxes). Containment test is unchanged (`fitz.Rect`).
- Granularity:
    * "block": current behaviour — drop a whole text block if its centre is in a
      region.
    * "line": for each line in each block, drop only that line if the LINE'S
      centre falls in a region; join the surviving lines. This protects captions
      that share a block with in-plot annotations.
  Keep both paths; select on `granularity`.
- Output contract unchanged: reading-order prose, paragraphs joined by "\n\n".

## Phase 2 — zotero_rag.py: wire extraction in, with table de-dup

In `extract_pages`, call `find_tables()` ONCE per page and use its result for BOTH
the table Markdown AND the prose exclusion:
- For each detected table: get `t.to_markdown()`. If non-empty, add it to the
  tables list AND add `fitz.Rect(t.bbox)` to a `table_rects` list. If a table's
  Markdown is empty, do NOT exclude its region (so we never drop table content
  with no Markdown replacement).
- Prose: `text = extract_page_text(page, extra_regions=table_rects)`.
- Preserve the existing per-page resilience (one bad page must not abort; keep the
  try/except structure and the "yield only if text or tables" guard).
- Keep the yield contract: `(page_no, text, tables_markdown)`.
Import `extract_page_text` from `figure_filter`.

## Phase 3 — cross-page chunking

Add `chunk_document(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]`
returning `(start_page, end_page, text)`:
- Split each page on blank lines (`\n\s*\n`), tag each paragraph with its page,
  flatten into one document-order stream. Hard-split oversized paragraphs with the
  existing `_hard_split(para, CHUNK_CHARS)`.
- Pack with the SAME budget/overlap logic as `chunk_page` (CHUNK_CHARS,
  CHUNK_OVERLAP, MIN_CHUNK_CHARS). Track the set of pages contributing to each
  chunk; tag the chunk with (min_page, max_page). Attribute the overlap carry to
  the new paragraph's page — the small mis-attribution of the carried tail is
  acceptable.

Schema: add `page_end: int` to `make_chunk_model`. Replace `_record` with
`_record_range(doc, page_start, page_end, suffix, text)` setting `page=page_start`
and `page_end=page_end`.

`ingest` loop: collect `pages = [(page_no, text), ...]` and accumulate table
records per page SEPARATELY (tables must NOT go through `chunk_document`). Then:
  - prose chunks  → `_record_range(doc, p_start, p_end, f"c{cidx}", chunk_text)`
  - table records → `_record_range(doc, page_no, page_no, f"{page_no}:t{tidx}:{sidx}", "Table:\n"+piece)`
Run `split_oversized_records` on the combined list as today. Chunk ids change
(prose now `c{n}`, doc-global); fine under rebuild.

## Phase 4 — diversity-aware selection

Add `select_diverse(candidates, scores, top_k, lambda_=MMR_LAMBDA,
per_doc_cap=PER_DOC_CAP) -> list[dict]`:
- Vectors are in `candidates` (LanceDB `.to_list()` returns the `vector` column)
  and are L2-normalized, so cosine == dot product.
- Min-max normalize `scores` to [0,1] for scale-stable mixing.
- Greedy MMR: relevance from `scores`; redundancy = max dot-product to ALREADY-
  SELECTED chunks FROM THE SAME doc_id ONLY (cross-document corroboration is NOT
  penalized — two papers stating the same fact should both survive).
- Soft per-doc cap: first pass enforces `per_doc_cap`; if it can't fill `top_k`,
  a second pass admits overflow so `top_k` is always returned.

Wire into `retrieve`:
  candidates = table.search(query_vector).limit(RERANK_CANDIDATES).to_list()
  if SELECT_DIVERSE and reranker available:
      scores = cross-encoder scores for ALL candidates
      return select_diverse(candidates, scores, top_k)
  else:
      return rerank(question, candidates, top_k)   # existing fallback path
Factor the cross-encoder scoring so `rerank` and the diverse path share it. If the
reranker is disabled or fails to load/predict, fall back to `candidates[:top_k]`
(current behaviour). Keep USE_RERANKER honoured.

## Phase 5 — multimodal + citations made range-aware

- `collect_page_images`: render each unique page across `[hit["page"],
  hit["page_end"]]` (not just `hit["page"]`), preserve order, cap at MAX_IMAGES.
- `_build_context`: cite `p.{page}` when `page == page_end`, else `pp.{page}-{page_end}`.
  Apply to both the context blocks and the source list.

## Validation (do not run against the user's library)
- Syntax/import check: `python -c "import ast,sys; ast.parse(open(f).read())"` for
  both files; confirm `figure_filter` imports cleanly.
- Leave a note that the user must run `python zotero_rag.py rebuild` (schema +
  chunking changed) and then spot-check on figure/equation/table-heavy papers:
  equations survive in prose chunks, figure legend/axis text is gone, captions
  kept, each table appears once (Markdown, not duplicated in prose), a page-
  straddling idea lands in one chunk, and repeated-query results spread across or
  de-dup within papers.

## Invariants
- Do not change the `extract_pages` yield shape or break the per-page error
  handling.
- Tables never go through `chunk_document`.
- `MAX_EMBED_CHARS` guard via `split_oversized_records` still runs last.
- No re-ingest logic changes beyond the schema field and record builder.