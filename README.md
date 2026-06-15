# Zotero RAG

Local retrieval-augmented generation over your Zotero PDF library. It indexes the PDFs in your Zotero storage, retrieves the most relevant passages for a question, reranks them with a cross-encoder, and generates a grounded answer with inline `[n]` citations back to the exact paper and page.

Embedding, reranking, and query rewriting always run locally on [Ollama](https://ollama.com). Only the final generation step contacts a model that may be remote — and the shipped default routes it through the LBNL Cborg gateway with page images attached (`GEN_PROVIDER="cborg"`, `MULTIMODAL=True`). Set `GEN_PROVIDER="ollama"` for a fully-local, offline setup.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model setup](#model-setup)
- [Quick start](#quick-start)
- [CLI commands](#cli-commands)
- [Configuration](#configuration)
- [Generation providers](#generation-providers)
- [Storage layout](#storage-layout)
- [Switching embedders / rerankers (A/B)](#switching-embedders--rerankers-ab)
- [Troubleshooting](#troubleshooting)
- [Notes and limitations](#notes-and-limitations)

## How it works

The pipeline runs in seven stages:

1. **Discovery** — walks `~/Zotero/storage` for PDFs and reads each paper's title/year from `zotero.sqlite` (opened read-only, so it's safe while Zotero is running), falling back to the filename when metadata is missing.
2. **Extraction** — figure-aware per-page text via PyMuPDF: text living inside large figures (axis labels, legends, in-plot annotations) is dropped, while prose, captions, and display equations are kept. Tables are detected once per page, rendered to Markdown, and their region is excluded from the prose so a table isn't also captured as garbled text.
3. **Chunking** — overlapping, paragraph-aware chunks (~350 tokens) packed *across* page breaks, so an idea that straddles a page boundary stays in one chunk. Each chunk records its page span, so citations point at a single page (`p.N`) or a range (`pp.N-M`).
4. **Embedding** — chunks become vectors via a selectable Ollama embedding model.
5. **Storage** — LanceDB, with **one table per embedder** (vectors from different models aren't comparable), keyed by model + dimension.
6. **Retrieval** — embed the query, pull the 32 nearest chunks, and score them all with a cross-encoder. Chunks scoring below a **relevance floor** are dropped — so a question your library can't support returns nothing (and the pipeline says so plainly, skipping generation entirely) rather than answering from irrelevant passages — while survivors that are only weakly matched are flagged *marginal* and carried through to the citations. Duplicate copies of the same work — same title under different DOIs, which Zotero often leaves unmerged — are then collapsed to a single source, so a paper indexed twice doesn't crowd out other papers (`TITLE_DEDUP`; audit them with `stats --duplicates`). Selection is **diversity-aware** by default (MMR): it trades relevance against redundancy *within the same paper* and applies a soft per-paper cap, so a single document can't dominate the context — while passages from *different* papers can still corroborate one another. Set `SELECT_DIVERSE=False` for plain top-k rerank ordering.
7. **Generation** — a grounded answer with inline `[n]` citations, via Ollama or a remote provider. Sources flagged *marginal relevance* are marked in the citation list and the model is told to lean on the stronger sources; if **every** retrieved source is marginal, the answer is prefixed with a low-confidence caveat. With `MULTIMODAL` enabled, the retrieved pages are rendered and attached so a vision model can read figures and tables directly.

Two query modes are available: a one-shot `query` with no memory, and a conversational `chat` with history-aware query rewriting.

## Requirements

- **Python 3.12** (the version this is developed and tested against; 3.10+ likely works but is unverified).
- **[Ollama](https://ollama.com)** installed and running — serves all local embedding, generation, and query-rewriting models.
- **Apple Silicon (MPS) recommended** for the reranker. It automatically falls back to CPU if MPS isn't available or fails.
- **Zotero** with attachments stored locally (the default "Linked/Stored" file storage under `~/Zotero/storage`).
- Disk space for the vector index (grows with library size) under `~/.cache/zotero_rag`.
- Optional, only for remote generation: an `ANTHROPIC_API_KEY` (Anthropic) or `CBORG_API_KEY` (Cborg).

## Installation

A dedicated virtual environment is recommended:

```bash
conda create -n zotero-rag python=3.12 && conda activate zotero-rag
pip install -r requirements.txt
```

If you don't already have a `requirements.txt`, create one with:

```text
ollama>=0.4
lancedb>=0.33.0
pymupdf>=1.24
pymupdf4llm>=1.27.2
tqdm>=4.66
numpy>=1.26
sentence-transformers>=5.0
transformers>=4.51.0
anthropic>=0.40
openai>=1.0
```

Notes on the pins:
- `lancedb>=0.33.0` guarantees `Table.optimize(cleanup_older_than=...)` and the `merge_insert` upsert builder used for crash-safe ingest.
- `sentence-transformers>=5.0` and `transformers>=4.51.0` are required for the Qwen3-Reranker option to tokenize/score correctly.
- `anthropic>=0.40` is only needed for the `anthropic` generation provider, and `openai>=1.0` only for the `cborg` provider. Both are imported lazily, so neither is required for the default fully-local setup.

## Model setup

Local models are served by Ollama and must be pulled once. The reranker is **not** an Ollama model — it downloads automatically from HuggingFace on first use.

```bash
# Default embedder (verify the exact tag in your Ollama registry first)
ollama pull qwen3-embedding:0.6b

# Default local generation model (used when GEN_PROVIDER="ollama")
ollama pull qwen3.6:27b

# Query-rewriting model used in chat mode
ollama pull qwen3.5:9b
```

Optional, only if you switch to them in the config:

```bash
ollama pull bge-m3                # alternative embedder
ollama pull qwen3-embedding:4b    # alternative embedder (2560-dim, higher quality)
ollama pull gemma4:26b            # vision-capable model, only if you enable MULTIMODAL
```

The default reranker `BAAI/bge-reranker-v2-m3` (0.6B params, ~1.1 GB in fp16) downloads from HuggingFace the first time you run a query or chat. The `gte` and `qwen3` reranker alternatives download on demand if selected.

> **Verify the embedder tag.** `qwen3-embedding:0.6b` is the configured default, but the exact Ollama registry tag can vary. If the pull 404s, find the correct tag and update the `ollama_model` field in the `EMBEDDERS` dict in `rag/config.py`.

## Quick start

```bash
# 1. Index your library (incremental and safe to re-run)
python zotero_rag.py ingest

# 2. Ask a one-shot question
python zotero_rag.py query "How is the kSZ effect modelled in stacking analyses?"

# 3. Or start an interactive session
python zotero_rag.py chat
```

The first `ingest` embeds your whole library and may take a while; subsequent runs only process new papers.

## CLI commands

Invoke as `python zotero_rag.py <command> [options]`.

### `ingest`

Index PDFs that haven't been indexed yet for the **active** embedder. Incremental (skips already-ingested papers via a manifest), idempotent (re-ingesting a paper updates rather than duplicates its chunks), and crash-safe (an interrupt leaves a consistent, resumable state; a single unreadable PDF is skipped and retried next run rather than aborting the whole run).

```bash
python zotero_rag.py ingest
```

### `query`

Answer a single question with no memory. Returns the grounded answer followed by a numbered source list.

```bash
python zotero_rag.py query "your question" [--top-k N]
```

| Option | Default | Description |
|---|---|---|
| `question` | — | The question (positional, quote it). |
| `--top-k` | `8` | Number of selected chunks used as context. |

If no index exists yet for the active embedder, it ingests first automatically.

### `chat`

Interactive conversation with memory and history-aware query rewriting (follow-ups are rewritten into standalone search queries). Type your questions; use `/exit` or `/quit` to leave.

```bash
python zotero_rag.py chat
```

### `stats`

Show the active configuration and every embedder table on disk (document count, chunk count, last-used time), plus the table cap.

```bash
python zotero_rag.py stats [--duplicates]
```

| Option | Default | Description |
|---|---|---|
| `--duplicates` | off | Audit duplicate library entries (same title under multiple `doc_id`s) per table. |

`--duplicates` flags titles ingested twice (e.g. preprint/published pairs) and labels each with what the query-time deduplicator (`TITLE_DEDUP`) would do: **`would merge`** (within `TITLE_DEDUP_YEAR_WINDOW` years — the rows to eyeball for wrong-merges), **`kept separate`** (years too far apart), or **`mixed`**. The verdict is indicative (the live order varies per query). Default `stats` omits the audit.

### `keep-only`

Drop **all** embedder tables except one — useful for collapsing to a single embedder once you've settled on one. Guards against a typo'd id (aborts instead of dropping everything) and prompts for confirmation.

```bash
python zotero_rag.py keep-only [embedder-id] [--yes]
```

| Option | Default | Description |
|---|---|---|
| `embedder-id` | active embedder | Which embedder table to keep (e.g. `qwen3-0.6b-1024`). |
| `--yes` | off | Skip the confirmation prompt. |

### `rebuild`

Drop the **active** embedder's table and manifest, then re-ingest the whole library from scratch. Leaves any other embedder tables (from A/B comparisons) intact. Prompts for confirmation.

```bash
python zotero_rag.py rebuild [--yes]
```

| Option | Default | Description |
|---|---|---|
| `--yes` | off | Skip the confirmation prompt. |

> For a complete wipe of **all** embedder tables, manifests, and the registry, delete the cache directory instead: `rm -rf ~/.cache/zotero_rag`.

## Configuration

Most knobs live in `rag/config.py` (including the figure-detection tunables — see the note below the table). The most important ones:

| Setting | Default | Options / notes |
|---|---|---|
| `EMBEDDER` | `qwen3-0.6b` | `qwen3-0.6b` (1024-dim) \| `bge-m3` (1024-dim) \| `qwen3-4b` (2560-dim). One-line A/B switch; changing it re-ingests into that embedder's own table. |
| `INSTRUCTION_TASK` | research-retrieval prompt | Prepended to **queries only** for instruction-aware (Qwen) embedders; `bge-m3` is symmetric and uses no instruction. |
| `MRL_TRUNCATE_4B` | `False` | Truncate the 4B embedder to 1024-dim for schema parity (small quality cost). |
| `RERANKER` | `bge` | `bge` (`BAAI/bge-reranker-v2-m3`) \| `gte` (`Alibaba-NLP/gte-reranker-modernbert-base`) \| `qwen3` (`Qwen/Qwen3-Reranker-0.6B`, slower). One-line switch; no re-ingest needed. |
| `USE_RERANKER` | `True` | Disable to use raw vector order. |
| `RERANK_DEVICE` | `mps` | Auto-falls back to CPU on load/predict failure. |
| `RERANK_CANDIDATES` | `32` | First-stage recall width before reranking. |
| `TOP_K` | `8` | Chunks selected and passed to the generator. |
| `SELECT_DIVERSE` | `True` | Diversity-aware final selection (MMR) over the scored candidates. `False` = plain top-k rerank order. |
| `MMR_LAMBDA` | `0.7` | Relevance vs within-paper redundancy trade-off (`1.0` = pure relevance). |
| `PER_DOC_CAP` | `3` | Soft cap on chunks selected from any one paper. |
| `TITLE_DEDUP` | `True` | Collapse duplicate copies of one work (same title, different DOI — which Zotero leaves unmerged) at query time, keeping the highest-scored copy. No re-ingest; audit with `stats --duplicates`. |
| `TITLE_DEDUP_YEAR_WINDOW` | `1` | Max year gap for two same-title copies to count as one work; `None` matches on the normalized title alone. |
| `GEN_PROVIDER` | `cborg` | `ollama` (local) \| `anthropic` (Claude API) \| `cborg` (LBNL gateway). |
| `GEN_MODEL` | `qwen3.6:27b` | Ollama generation model (used when `GEN_PROVIDER="ollama"`). For `MULTIMODAL` it must be vision-capable and non-MLX (e.g. `gemma4:26b`). |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Used when `GEN_PROVIDER="anthropic"`. |
| `CBORG_MODEL` | `anthropic/claude-sonnet` | Cborg alias; used when `GEN_PROVIDER="cborg"`. See [Generation providers](#generation-providers). |
| `CBORG_BASE_URL` | `https://api.cborg.lbl.gov` | Cborg API endpoint. |
| `REWRITE_MODEL` | `qwen3.5:9b` | Local Ollama model for chat query rewriting (always local, regardless of `GEN_PROVIDER`). |
| `MULTIMODAL` | `True` | Render retrieved pages to images and attach them to a vision generation model. |
| `IMAGE_DPI` / `MAX_IMAGES` | `150` / `3` | Page-image render resolution and cap. |
| `CHUNK_CHARS` | `1400` | Target chunk size; over-long paragraphs and tables are split below this. |
| `CHUNK_OVERLAP` | `250` | Overlap between adjacent chunks. |
| `MAX_EMBEDDER_TABLES` | `2` | Max embedder tables kept at once (LRU eviction). |
| `VERSION_RETENTION` | `7 days` | LanceDB version/fragment retention reclaimed on each ingest. Never set to `0`. |
| `ZOTERO_DIR` | `~/Zotero` | Zotero data directory. |
| `INDEX_DIR` | `~/.cache/zotero_rag` | Vector index, manifests, and registry. |

A few **figure-detection** tunables also live in `rag/config.py` (consumed by `rag/figure_filter.py`): `FIGURE_MIN_AREA_FRAC` (the area gate that separates figures from display equations), `FIGURE_PAD` (region inflation, in points), and `FIGURE_GRANULARITY` (`"line"` drops only in-figure lines, protecting captions that share a text block with plot annotations; `"block"` drops whole blocks).

**Relevance thresholds (per reranker).** Each entry in the `RERANKERS` dict carries an optional `min_score` (hard floor) and `soft_score` (marginal cut), both on the cross-encoder's **sigmoid `[0, 1]` scale**. Chunks below `min_score` are dropped (a query with nothing above the floor returns no answer and skips generation); survivors below `soft_score` are kept but flagged *marginal relevance* in the citations and the model context, and an all-marginal result is captioned low-confidence. Raw cross-encoder logits are unbounded and model-specific, so these are only meaningful with `sigmoid=True` — the default `bge` (`min_score=0.075`, `soft_score=0.5`) and `qwen3` enable it, while `gte` leaves both unset (`None`), disabling gating and preserving plain ranking. The values are calibrated against a library's own score distribution; retune them with `tests/probe_reranker_sigmoid.py`.

## Generation providers

Set `GEN_PROVIDER` to choose where answers are generated. Embedding and reranking are unaffected — they're always local.

### `ollama` (fully local)

Uses `GEN_MODEL` (default `qwen3.6:27b`). No API key, no network. This is the only provider that works fully offline.

### `anthropic` (Claude API)

Uses `ANTHROPIC_MODEL` via the official Anthropic API. Requires an API key in the environment:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### `cborg` (LBNL gateway, shipped default)

Routes generation through Berkeley Lab's [Cborg](https://cborg.lbl.gov) gateway (free for `@lbl.gov`, `@es.net`, and `@nersc.gov` accounts). Requires a key in the environment:

```bash
export CBORG_API_KEY="..."
```

Cborg is an OpenAI-compatible [LiteLLM](https://litellm.ai) proxy, so the script talks to it with the OpenAI SDK (`chat.completions`) against `https://api.cborg.lbl.gov`. Model names are provider-prefixed Cborg aliases — the default is `anthropic/claude-sonnet` (`CBORG_MODEL`); query your account's model list to see what else is available. The `openai` package is required for this provider (imported lazily, only when `GEN_PROVIDER="cborg"`).

> **Multimodal over Cborg works for vision models.** With `MULTIMODAL` enabled the script attaches OpenAI-style `image_url` blocks, which LiteLLM transcodes into the backend model's native image format. This is verified to reach the default `anthropic/claude-sonnet` (`tests/probe_cborg_multimodal.py` confirms the model reads an in-image-only canary and the prompt-token count rises accordingly). Images only land if `CBORG_MODEL` is vision-capable — a text-only alias silently drops them — so the script prints a reminder to stderr noting the active model. Re-run the probe after changing `CBORG_MODEL` to confirm.

## Storage layout

Everything lives under `INDEX_DIR` (`~/.cache/zotero_rag`):

```
~/.cache/zotero_rag/
└── lancedb/
    ├── chunks__qwen3-0.6b-1024.lance/    # one table per embedder (model+dim)
    ├── chunks__bge-m3-1024.lance/
    ├── ingested__qwen3-0.6b-1024.json    # per-embedder ingest manifest
    ├── ingested__bge-m3-1024.json
    └── embedder_registry.json            # last-used timestamps for LRU eviction
```

- **One table per embedder.** Vectors from different models aren't comparable, so each embedder gets its own table keyed by `model+dim` (e.g. `qwen3-0.6b-1024`).
- **Bounded growth.** At most `MAX_EMBEDDER_TABLES` (default 2) coexist. When you switch embedders, the new table is built first, then the least-recently-used previous table is evicted — so a crash mid-switch never drops your working table.
- **Cruft reclaimed.** Each ingest ends with `table.optimize()`, compacting fragments and pruning versions older than `VERSION_RETENTION`.

## Switching embedders / rerankers (A/B)

**Reranker:** change `RERANKER` and re-run any query. No re-ingest — reranking only re-reads candidate text at query time.

**Embedder:** change `EMBEDDER` and run `ingest` (or any query, which ingests automatically if needed). The first run for a new embedder re-embeds the library into that embedder's own table; the previous embedder's table is kept as a comparison slot (up to the cap of 2), so you can switch back and forth without re-ingesting each time. Use `stats` to see what's stored and `keep-only` to collapse to one once you've decided.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ollama pull qwen3-embedding:0.6b` 404s | The registry tag differs. Find the correct tag and update the `ollama_model` in the `EMBEDDERS` dict. |
| `MuPDF error: ... could not parse color space (NNN 0 R)` | Benign warning from a malformed colour space in some PDFs; text extraction is unaffected. Silence with `fitz.TOOLS.mupdf_display_errors(False)`. |
| `Consider using the pymupdf_layout package ...` | Benign informational notice from `find_tables()` advertising an optional add-on. Not an error; it does **not** silence via `mupdf_display_errors` (it prints to stdout). Ignore it, or wrap the call in a stdout redirect. |
| `warning: skipping <pdf> (ResponseError ... embeddings ... EOF)` | An oversized embedding input crashed the Ollama runner. Fixed by the chunk-size caps; the skipped paper retries on the next `ingest`. If it recurs, check your Ollama version (some 0.13.x releases regressed embedding batch handling). |
| Citations show filenames instead of titles | The Zotero SQLite metadata lookup fell back to the filename. Check that `zotero.sqlite` is present and attachments are stored locally. |
| A scanned PDF isn't indexed | No text layer to extract (no OCR). Such PDFs are skipped. |
| Reranker slow or warns about MPS | It auto-falls back to CPU. The `qwen3` reranker is autoregressive and slower than `bge`/`gte` by design. |

## Notes and limitations

- **Local-first.** Embedding, reranking, and query rewriting always run locally on Ollama/MPS. Only final generation can be remote.
- **Out-of-domain queries are gated.** Retrieval applies a per-reranker relevance floor on the cross-encoder's sigmoid score, so a question your library can't support returns "no sufficiently relevant content" (and skips the generation call) instead of answering from unrelated passages; weakly-matched sources are flagged *marginal* and an all-marginal answer is captioned low-confidence. The floor is only active for `sigmoid=True` rerankers and is calibrated per library (see [Configuration](#configuration)).
- **No OCR.** Scanned/image-only PDFs without a text layer are skipped.
- **Apple Silicon assumed for the reranker.** It runs on MPS with a CPU fallback; other accelerators aren't specially handled.
- **Ingest is incremental and resumable.** Re-running `ingest` only processes new or previously-failed papers; it won't duplicate existing chunks.
- **Switching the embedder requires a re-ingest** into that embedder's table (one-time cost). Switching the reranker does not.
- **One attachment, multiple PDFs.** A document's `doc_id` is its Zotero storage-folder key, and chunk ids are derived from it (`<doc_id>:c<n>` for prose chunks, `<doc_id>:<page>:t<tidx>:<sidx>` for tables). A folder normally holds one PDF, but if it holds several (e.g. supplementary files), the first (sorted) keeps the bare key and the rest get a `<key>__<stem>` suffix, so every PDF is indexed with distinct ids and none overwrites another. Note: if such a folder's first PDF is later removed, the bare key would attach to a different file on the next ingest — delete and rebuild that embedder's table if you reshuffle PDFs within a folder.
