# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local RAG pipeline over a Zotero library, organized as a `rag/` package with a thin
`zotero_rag.py` launcher (so `python zotero_rag.py <command>` still works). Flow: discover PDFs →
extract (figure-aware) → chunk across pages → embed → store in LanceDB → retrieve →
rerank / diversity-select → generate a cited answer. Everything runs locally by default via
Ollama; only the *generation* step can optionally be routed to a remote Claude-compatible API.

Package layout (`rag/`): `config.py` (all tunables), `library.py` (Zotero discovery/metadata),
`extraction.py` + `figure_filter.py` (PDF → figure-aware page text + tables, page images),
`chunking.py` (chunking + size guards), `embedding.py` (embedder identity + Ollama embedding),
`schema.py` (dynamic LanceDB model), `reranking.py` (cross-encoder), `selection.py` (MMR +
title canonicalization), `generation.py` (provider fan-out), `pipeline.py` (`RAGPipeline`:
LanceDB lifecycle + query path), `cli.py` (argparse). There is a `README.md` and ad-hoc test
scripts under `tests/`; no pytest suite.

## Environment & commands

Conda env `zotero-rag` (Python 3.12); the VS Code interpreter is pinned to it. Activate before
running:

```bash
conda activate zotero-rag
pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b      # default embedder; Ollama must be running

python zotero_rag.py ingest                       # index new PDFs for the active embedder
python zotero_rag.py query "your question" --top-k 6
python zotero_rag.py chat                          # interactive, history-aware
python zotero_rag.py stats [--duplicates]          # config + per-table counts; --duplicates: title-dup audit w/ merge/keep verdict (default omits)
python zotero_rag.py keep-only [embedder-id] --yes # collapse to one embedder table
python zotero_rag.py rebuild --yes                 # drop active table + manifest, re-ingest fresh
```

There is no lint/format/test tooling configured. Style is `black`-compatible, Google docstrings,
type hints on all functions (the global conventions apply).

## Configuration model

Almost all tuning is done by editing the constants in `rag/config.py` — there are no config files or
env-based settings flags. The intended workflow is literally a one-line edit:

- `EMBEDDER` — selects from the `EMBEDDERS` dict. **Switching re-ingests** into that embedder's own table.
- `RERANKER` — selects from `RERANKERS`. Switching needs **no** re-ingest.
- `GEN_PROVIDER` — `"ollama"` (local) | `"anthropic"` (Claude API) | `"cborg"` (LBNL gateway).
- `SELECT_DIVERSE` / `MMR_LAMBDA` / `PER_DOC_CAP` — diversity-aware final selection (MMR with a
  soft per-paper cap). Switching needs **no** re-ingest.
- `MULTIMODAL` — attach rendered page images to a vision generation model (retrieval stays text-only).

Secrets come from the environment, not code: `ANTHROPIC_API_KEY` for `anthropic`, `CBORG_API_KEY`
for `cborg`.

Figure-detection tunables (`FIGURE_MIN_AREA_FRAC`, `FIGURE_PAD`, `FIGURE_GRANULARITY`) live in
`rag/config.py` alongside the other tunables and are consumed by `rag/figure_filter.py`.

## Architecture invariants (read before editing)

These cross-cutting rules are the reason the code is shaped the way it is. Breaking one silently
corrupts the index or crashes the embedding runner.

- **One table per embedder.** Vectors from different models aren't comparable, so each embedder
  writes to `chunks__<embedder_id>` where `embedder_id = "<EMBEDDER>-<effective_dim>"` (e.g.
  `qwen3-0.6b-1024`). The dynamic schema is built per-dimension by `make_chunk_model(dim)` because
  `Vector(dim)` is fixed at class-definition time. `_verify_dim` guards against a model/table dim mismatch.

- **Embedding asymmetry.** Documents are embedded with no instruction; queries get an `Instruct:`
  prefix **only** for instruction-aware (Qwen) embedders. Vectors are truncated to `effective_dim()`
  and L2-normalised, so plain L2 search ranks identically to cosine (no metric is set at search time).

- **Figure-aware extraction, table de-dup.** Prose comes from `rag.figure_filter.extract_page_text`,
  which drops text whose centre lies inside a large graphic region (area-gated by
  `FIGURE_MIN_AREA_FRAC`, so display equations survive) or inside a detected table's bbox.
  `extract_pages` calls `find_tables()` ONCE per page and feeds the result to BOTH the Markdown
  chunk and the prose-exclusion regions; a table whose Markdown is empty is NOT excluded (so no
  content is dropped without a Markdown replacement). `get_text("dict")` is parsed once per page.

- **Cross-page chunking.** Prose is packed across page breaks by `chunk_document`; tables are kept
  per-page and **never** routed through it (packing would scramble their Markdown). Each chunk
  carries a `page`/`page_end` span — `page_end` is a schema field, so **changing the chunking
  requires a full `rebuild`** (there is no in-place migration; `_verify_dim` only guards the vector
  dim). Citations render `p.N` or `pp.N-M`.

- **Size guard chain.** A chunk larger than the embedding runner's batch crashes it (EOF). Three
  layers prevent this: `chunk_document` hard-splits over-long paragraphs, table Markdown is
  `_hard_split`, and `split_oversized_records` is the final net enforcing `MAX_EMBED_CHARS`. Don't
  loosen these without understanding the crash they prevent.

- **Diversity-aware retrieval.** `retrieve` cross-encoder scores all candidates ONCE via
  `_cross_encoder_scores`. With `TITLE_DEDUP`, `_canonicalize_by_title` then runs (after scoring,
  before selection): processing candidates in DESCENDING score order, it keeps one `doc_id` per work
  — anchoring each by (normalized title within `TITLE_DEDUP_YEAR_WINDOW` years) and dropping the
  lower-scored other copies of that same title/year. **The descending-score order is load-bearing**:
  the first copy seen anchors the work, so the wrong order would keep the worse copy. This is what
  catches the duplicates Zotero misses — same title, different DOI, which Zotero's DOI-mismatch veto
  leaves unmerged. Different normalized titles are NEVER merged, so cross-paper corroboration is
  preserved. With `SELECT_DIVERSE`, `select_diverse` then runs greedy MMR with redundancy penalised
  only WITHIN the same `doc_id` (cross-document corroboration is not penalised) plus a soft
  `PER_DOC_CAP` two-pass fill; otherwise the survivors are ranked by score and sliced. A disabled or
  failed reranker falls back to vector-search order (no dedup) in every path.

- **Idempotent, crash-safe ingest.** Records use deterministic ids (prose `<doc_id>:c<n>`, tables
  `<doc_id>:<page>:t<tidx>:<sidx>`) and are
  written with `merge_insert("id")` (upsert), **never** `table.add` (which double-counts on retry).
  The per-embedder manifest of completed `doc_id`s is saved incrementally and in a `finally`, so
  Ctrl-C resumes cleanly. A PDF that errors is left *unmarked* (retried next run); a PDF that opens
  but yields no text is marked done (so it isn't retried forever). Retrieval/optimize/eviction only
  run on normal completion.

- **Table retention.** At most `MAX_EMBEDDER_TABLES` coexist. Switching embedders builds the new
  table first, then `_enforce_cap` evicts the least-recently-used non-active one. A JSON
  `embedder_registry.json` tracks last-used times and is reconciled against on-disk tables (disk is
  ground truth) on every prepare. `VERSION_RETENTION` must never be 0 (breaks concurrent writers).

- **Zotero is read-only.** `zotero.sqlite` is opened `mode=ro&immutable=1` so the pipeline works
  while Zotero is running. Metadata lookup is best-effort: any failure falls back to the PDF filename.

- **Provider fan-out.** Three separate generation paths, all in `rag/generation.py` behind the
  `generate()` dispatcher. `anthropic` uses the Anthropic SDK
  (`_generate_anthropic`, Messages API). `cborg` (LBNL gateway) is an **OpenAI-compatible LiteLLM
  proxy**, so it uses the **OpenAI SDK** (`_generate_openai`, `chat.completions`) with a bearer token and
  custom `base_url`, and provider-prefixed model aliases (`anthropic/claude-sonnet`) — it is *not*
  Anthropic-API-compatible despite generating with Claude. Ollama has its own path. All three pass
  `GEN_TEMPERATURE`. Chat-mode query rewriting always runs locally via Ollama regardless of `GEN_PROVIDER`.

## Storage layout

Index lives under `~/.cache/zotero_rag/lancedb/`: `chunks__<id>.lance` tables,
`ingested__<id>.json` manifests, and `embedder_registry.json`. Source PDFs are read from
`~/Zotero/storage/*/*.pdf`. A full wipe is `rm -rf ~/.cache/zotero_rag`.
