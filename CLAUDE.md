# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file (`zotero_rag.py`) local RAG pipeline over a Zotero library: discover PDFs →
extract → chunk → embed → store in LanceDB → retrieve → rerank → generate a cited answer.
Everything runs locally by default via Ollama; only the *generation* step can optionally be
routed to a remote Claude-compatible API. There is no package, no README, and no test suite.

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
python zotero_rag.py stats                         # active config + per-table row/doc counts
python zotero_rag.py keep-only [embedder-id] --yes # collapse to one embedder table
python zotero_rag.py rebuild --yes                 # drop active table + manifest, re-ingest fresh
```

There is no lint/format/test tooling configured. Style is `black`-compatible, Google docstrings,
type hints on all functions (the global conventions apply).

## Configuration model

All tuning is done by editing the constants in the `=== Configuration ===` block at the top of
`zotero_rag.py` — there are no config files or env-based settings flags. The intended workflow is
literally a one-line edit:

- `EMBEDDER` — selects from the `EMBEDDERS` dict. **Switching re-ingests** into that embedder's own table.
- `RERANKER` — selects from `RERANKERS`. Switching needs **no** re-ingest.
- `GEN_PROVIDER` — `"ollama"` (local) | `"anthropic"` (Claude API) | `"cborg"` (LBNL gateway).
- `MULTIMODAL` — attach rendered page images to a vision generation model (retrieval stays text-only).

Secrets come from the environment, not code: `ANTHROPIC_API_KEY` for `anthropic`, `CBORG_API_KEY`
for `cborg`.

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

- **Size guard chain.** A chunk larger than the embedding runner's batch crashes it (EOF). Three
  layers prevent this: `chunk_page` hard-splits over-long paragraphs, table Markdown is `_hard_split`,
  and `split_oversized_records` is the final net enforcing `MAX_EMBED_CHARS`. Don't loosen these
  without understanding the crash they prevent.

- **Idempotent, crash-safe ingest.** Records use deterministic ids (`<doc_id>:<page>:<idx>`) and are
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

- **Provider fan-out.** Three separate generation paths. `anthropic` uses the Anthropic SDK
  (`_generate_messages_api`). `cborg` (LBNL gateway) is an **OpenAI-compatible LiteLLM proxy**, so it
  uses the **OpenAI SDK** (`_generate_openai_compatible`, `chat.completions`) with a bearer token and
  custom `base_url`, and provider-prefixed model aliases (`anthropic/claude-sonnet`) — it is *not*
  Anthropic-API-compatible despite generating with Claude. Ollama has its own path. All three pass
  `GEN_TEMPERATURE`. Chat-mode query rewriting always runs locally via Ollama regardless of `GEN_PROVIDER`.

## Storage layout

Index lives under `~/.cache/zotero_rag/lancedb/`: `chunks__<id>.lance` tables,
`ingested__<id>.json` manifests, and `embedder_registry.json`. Source PDFs are read from
`~/Zotero/storage/*/*.pdf`. A full wipe is `rm -rf ~/.cache/zotero_rag`.
