"""Local RAG over a Zotero library: retrieval + reranking + chat, on Ollama.

Everything runs locally by default. Generation can optionally be routed to the
Claude API (the only role that can be remote -- embedding and reranking always
run locally, since Claude is not an embedding or reranking model).

PIPELINE (stage by stage):
    1. Discovery   walk ~/Zotero/storage for PDFs; read titles/years from
                   zotero.sqlite (read-only), falling back to the filename.
    2. Extraction  text per page with PyMuPDF; tables -> Markdown chunks too.
    3. Chunking    overlapping ~350-token chunks; per-page -> exact page cites.
    4. Embedding   chunks -> vectors via an Ollama embedding model (selectable).
    5. Storage     LanceDB, ONE table per embedder (vectors of different models
                   are not comparable), keyed by model+dim.
    6. Retrieve    embed query, pull 24 nearest, cross-encoder rerank to top-k.
    7. Generate    grounded answer with [n] citations via Ollama or Claude.
                   If MULTIMODAL, retrieved pages are rendered and attached so a
                   vision model can read figures/tables.

MODES:
    query  one-shot question, no memory.
    chat   conversation with memory + history-aware query rewriting.

KEY CONFIG KNOBS (all near the top):
    EMBEDDER       which embedding model (one-line A/B; switching re-ingests
                   into that embedder's own table).
    RERANKER       which cross-encoder (one-line A/B; no re-ingest).
    GEN_PROVIDER   "ollama", "anthropic", or "cborg" for the answer model.
    MULTIMODAL     attach page images to a vision generation model.
    MAX_EMBEDDER_TABLES / VERSION_RETENTION   storage-retention controls.

STORAGE RETENTION:
    * Each embedder writes to chunks__<id> (id = model+dim, e.g. qwen3-0.6b-1024).
    * At most MAX_EMBEDDER_TABLES coexist; on a switch the new table is built
      first, then the least-recently-used previous table is evicted.
    * `keep-only` collapses to a single embedder on demand.
    * Each ingest ends with table.optimize() to reclaim version/fragment cruft.

SETUP (dedicated env recommended):
    conda create -n zotero-rag python=3.12 && conda activate zotero-rag
    pip install -r requirements.txt
    ollama pull qwen3-embedding:0.6b      # default embedder (verify exact tag)

USAGE:
    python zotero_rag.py ingest
    python zotero_rag.py query "your question"
    python zotero_rag.py chat
    python zotero_rag.py stats
    python zotero_rag.py keep-only [embedder-id]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import lancedb
import numpy as np
import ollama
from lancedb.pydantic import LanceModel, Vector
from tqdm import tqdm

# === Configuration ==========================================================

ZOTERO_DIR = Path.home() / "Zotero"
INDEX_DIR = Path.home() / ".cache" / "zotero_rag"


@dataclass(frozen=True)
class EmbedderSpec:
    """An embedding model option.

    Attributes:
        ollama_model: Ollama model tag to pull/serve.
        dim: Native embedding dimension.
        instruct: Whether the model is instruction-aware (prefix queries only).
    """

    ollama_model: str
    dim: int
    instruct: bool


# Selectable embedders. Switching is a one-line change to EMBEDDER below; each
# embedder uses its OWN table, so a switch triggers a one-time re-ingest.
EMBEDDERS: dict[str, EmbedderSpec] = {
    "qwen3-0.6b": EmbedderSpec("qwen3-embedding:0.6b", 1024, instruct=True),
    "bge-m3": EmbedderSpec("bge-m3", 1024, instruct=False),
    "qwen3-4b": EmbedderSpec("qwen3-embedding:4b", 2560, instruct=True),
}
EMBEDDER = "qwen3-0.6b"  # <-- one-line A/B switch: "qwen3-0.6b" | "bge-m3" | "qwen3-4b"

# Instruction prepended to QUERIES ONLY for instruction-aware (Qwen) embedders.
# Documents are embedded with no instruction.
INSTRUCTION_TASK = (
    "Given a research question, retrieve passages from scientific papers that answer it."
)

# Optional MRL truncation of the 2560-dim qwen3-4b to 1024 (schema parity only;
# NOT for storage). Default OFF -> native 2560. Toggling this yields a distinct
# embedder-id/table (qwen3-4b-1024 vs qwen3-4b-2560).
MRL_TRUNCATE_4B = False

EMBED_NUM_CTX = 8192
EMBED_BATCH = 32


@dataclass(frozen=True)
class RerankerSpec:
    """A cross-encoder reranker option (all run via sentence-transformers).

    Attributes:
        model: HuggingFace model id.
        trust_remote_code: Pass through to CrossEncoder (needed for gte).
        sigmoid: Apply Sigmoid activation for 0-1 scores (Qwen3-Reranker).
    """

    model: str
    trust_remote_code: bool
    sigmoid: bool


# Selectable rerankers. All three load through the same sentence-transformers
# CrossEncoder interface (per-model flags below). Reranking runs on MPS, NOT
# Ollama (which has no rerank endpoint). Switching needs no re-ingest.
RERANKERS: dict[str, RerankerSpec] = {
    "bge": RerankerSpec("BAAI/bge-reranker-v2-m3", trust_remote_code=False, sigmoid=False),
    "gte": RerankerSpec("Alibaba-NLP/gte-reranker-modernbert-base", trust_remote_code=True, sigmoid=False),
    # Qwen3-Reranker is autoregressive -> higher per-pair latency than the BERT-based two.
    "qwen3": RerankerSpec("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=False, sigmoid=True),
}
RERANKER = "bge"  # <-- one-line A/B switch: "bge" | "gte" | "qwen3"

USE_RERANKER = True
RERANK_DEVICE = "mps"  # falls back to CPU automatically if MPS load fails
RERANK_CANDIDATES = 24  # wide first-stage recall, before reranking
TOP_K = 6               # final chunks passed to the generator

# Generation backend.
GEN_PROVIDER = "ollama"  # "ollama" (local) | "anthropic" (Claude API) | "cborg" (LBNL gateway)
GEN_MODEL = "qwen3.6:35b"  # Ollama model. For MULTIMODAL must be vision + NON-MLX.
GEN_NUM_CTX = 8192
GEN_TEMPERATURE = 0.2
ANTHROPIC_MODEL = "claude-sonnet-4-6"  # used only when GEN_PROVIDER == "anthropic"
ANTHROPIC_MAX_TOKENS = 2048            # ANTHROPIC_API_KEY must be set in the env

# CBORG (LBNL) is an OpenAI-compatible gateway built on LiteLLM: it exposes the
# OpenAI chat-completions API (NOT the Anthropic Messages API), with bearer-token
# auth against a custom base URL. Used only when GEN_PROVIDER == "cborg". The
# token is read from $CBORG_API_KEY (exported in ~/.zshrc). Model names are
# provider-prefixed CBORG aliases (e.g. "anthropic/claude-sonnet"), not bare
# Anthropic ids. NOTE: CBORG's support for image content is unverified, so
# MULTIMODAL may or may not work over this provider (a warning is printed if used).
CBORG_MODEL = "anthropic/claude-sonnet"  # provider-prefixed CBORG alias (cf. anthropic/claude-haiku)
CBORG_MAX_TOKENS = 2048
CBORG_BASE_URL = "https://api.cborg.lbl.gov"

# Chat-mode query rewriting always runs locally via Ollama (small, bounded task),
# regardless of GEN_PROVIDER. Change this to taste.
REWRITE_MODEL = "qwen3.5:9b"
MAX_HISTORY_MESSAGES = 6

# Multimodal generation (attach rendered page images). OFF by default; retrieval
# stays text-based regardless.
MULTIMODAL = False
IMAGE_DPI = 150
MAX_IMAGES = 3

# Chunking.
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 250
MIN_CHUNK_CHARS = 120
# Hard ceiling on a single embedding input. A chunk larger than the embedding
# runner's batch crashes it ("unable to fit entire input in a batch" -> EOF), so
# no record may exceed this. Set just above the natural chunk ceiling
# (cap + overlap + join) so the guard never splits normal chunks, only
# pathologically large ones.
MAX_EMBED_CHARS = CHUNK_CHARS + CHUNK_OVERLAP + 50

# Storage retention.
MAX_EMBEDDER_TABLES = 2
VERSION_RETENTION = timedelta(days=7)  # never set to 0 (breaks concurrent writers)


# === Embedder identity / dimension ==========================================


def _embedder_spec() -> EmbedderSpec:
    """Return the spec for the active embedder."""
    return EMBEDDERS[EMBEDDER]


def effective_dim() -> int:
    """Return the stored vector dimension for the active embedder.

    Equals the model's native dim, except the qwen3-4b is truncated to 1024 when
    ``MRL_TRUNCATE_4B`` is set.

    Returns:
        The embedding dimension actually written to LanceDB.
    """
    if EMBEDDER == "qwen3-4b" and MRL_TRUNCATE_4B:
        return 1024
    return _embedder_spec().dim


def embedder_id() -> str:
    """Return the identity key for the active embedder, e.g. ``qwen3-0.6b-1024``."""
    return f"{EMBEDDER}-{effective_dim()}"


# === Embedding (Ollama) =====================================================


def _embed_raw(texts: list[str]) -> list[list[float]]:
    """Embed texts via the active Ollama embedding model (no pre/post-processing).

    Args:
        texts: Texts to embed.

    Returns:
        Raw embedding vectors, one per input.
    """
    model = _embedder_spec().ollama_model
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        resp = ollama.embed(model=model, input=batch, options={"num_ctx": EMBED_NUM_CTX})
        embs = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
        vectors.extend([list(e) for e in embs])
    return vectors


def _postprocess(vectors: list[list[float]]) -> list[list[float]]:
    """Truncate to the effective dimension and L2-normalise (float32).

    Normalising means the default L2 distance ranks identically to cosine, so no
    explicit metric is needed at search time.

    Args:
        vectors: Raw embedding vectors.

    Returns:
        Truncated, unit-normalised float32 vectors as plain lists.
    """
    dim = effective_dim()
    arr = np.asarray(vectors, dtype=np.float32)[:, :dim]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk/document texts (no instruction prefix).

    Args:
        texts: Document texts.

    Returns:
        Processed embedding vectors.
    """
    return _postprocess(_embed_raw(texts))


def embed_query(text: str) -> list[float]:
    """Embed a query, prepending the instruction for instruction-aware embedders.

    Args:
        text: The query string.

    Returns:
        A single processed embedding vector.
    """
    if _embedder_spec().instruct:
        text = f"Instruct: {INSTRUCTION_TASK}\nQuery: {text}"
    return _postprocess(_embed_raw([text]))[0]


# === Reranking (sentence-transformers / MPS) ================================


@lru_cache(maxsize=1)
def _get_reranker():
    """Load the active cross-encoder once, with a CPU fallback and smoke-test.

    Returns:
        A loaded ``sentence_transformers.CrossEncoder``.
    """
    import torch
    from sentence_transformers import CrossEncoder

    spec = RERANKERS[RERANKER]
    kwargs: dict = {"trust_remote_code": spec.trust_remote_code}
    if spec.sigmoid:
        kwargs["activation_fn"] = torch.nn.Sigmoid()
    # Smoke-test inside the try so a predict-time failure on MPS (e.g. an
    # unsupported op, the known risk for gte-reranker-modernbert-base) also
    # falls back to CPU; only a CPU failure propagates.
    try:
        model = CrossEncoder(spec.model, device=RERANK_DEVICE, **kwargs)
        model.predict([("warmup query", "warmup passage")])
    except Exception:
        model = CrossEncoder(spec.model, device="cpu", **kwargs)
        model.predict([("warmup query", "warmup passage")])
    return model


def rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-order candidates by cross-encoder relevance, with graceful fallback.

    Args:
        question: The query (already rewritten, in chat mode).
        hits: Candidate chunk dicts from vector search.
        top_k: Number of chunks to keep.

    Returns:
        The ``top_k`` most relevant chunks. Falls back to vector order if the
        reranker is disabled or fails to load.
    """
    if not USE_RERANKER or not hits:
        return hits[:top_k]
    try:
        model = _get_reranker()
    except Exception:
        return hits[:top_k]
    scores = model.predict([(question, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [hit for hit, _ in ranked[:top_k]]


# === Zotero discovery =======================================================


@dataclass
class Document:
    """A source PDF drawn from the Zotero library.

    Attributes:
        doc_id: Zotero attachment storage key (stable, unique per attachment).
        title: Best-effort human-readable title.
        year: Publication year as a string, or "".
        pdf_path: Absolute path to the PDF on disk.
    """

    doc_id: str
    title: str
    year: str
    pdf_path: Path


def _field_values(cur: sqlite3.Cursor, field_name: str) -> dict[int, str]:
    """Return ``{itemID: value}`` for one Zotero metadata field.

    Args:
        cur: Open cursor on the Zotero database.
        field_name: Zotero field name, e.g. ``"title"`` or ``"date"``.

    Returns:
        Mapping from item id to value.
    """
    rows = cur.execute(
        """
        SELECT d.itemID, v.value
        FROM itemData d
        JOIN itemDataValues v ON v.valueID = d.valueID
        JOIN fields f ON f.fieldID = d.fieldID
        WHERE f.fieldName = ?
        """,
        (field_name,),
    ).fetchall()
    return {item_id: value for item_id, value in rows}


def load_zotero_metadata(zotero_dir: Path) -> dict[str, tuple[str, str]]:
    """Map each attachment storage key to ``(title, year)``, best-effort.

    Opens ``zotero.sqlite`` read-only/immutable so it works while Zotero runs.
    Any failure degrades to an empty map; callers fall back to the filename.

    Args:
        zotero_dir: Path to the Zotero data directory.

    Returns:
        ``{attachment_key: (title, year)}``. May be empty.
    """
    db_path = zotero_dir / "zotero.sqlite"
    if not db_path.exists():
        return {}
    uri = f"file:{db_path}?mode=ro&immutable=1"
    out: dict[str, tuple[str, str]] = {}
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return {}
    try:
        cur = con.cursor()
        titles = _field_values(cur, "title")
        dates = _field_values(cur, "date")
        attachments = cur.execute(
            """
            SELECT i.key, a.parentItemID
            FROM itemAttachments a
            JOIN items i ON i.itemID = a.itemID
            WHERE a.parentItemID IS NOT NULL
            """
        ).fetchall()
        for key, parent_id in attachments:
            title = titles.get(parent_id, "")
            raw_date = dates.get(parent_id, "")
            year_match = re.search(r"\b(\d{4})\b", raw_date or "")
            if title:
                out[key] = (title, year_match.group(1) if year_match else "")
    except sqlite3.Error:
        return {}
    finally:
        con.close()  # always release the handle, even on KeyboardInterrupt
    return out


def discover_documents(zotero_dir: Path) -> list[Document]:
    """Find every stored PDF in the Zotero library with best-effort metadata.

    A Zotero storage folder is keyed by an attachment key and normally holds a
    single PDF, but can occasionally hold several (e.g. supplementary files). All
    PDFs are indexed; to keep ``doc_id`` (and thus the chunk ids) unique without
    re-ingesting the single-PDF common case, the first PDF in a multi-PDF folder
    keeps the bare key as its ``doc_id`` and the rest get a ``<key>__<stem>``
    suffix.

    Args:
        zotero_dir: Path to the Zotero data directory.

    Returns:
        One :class:`Document` per PDF under ``storage/``, each with a unique
        ``doc_id``.
    """
    storage = zotero_dir / "storage"
    if not storage.is_dir():
        raise FileNotFoundError(
            f"No Zotero storage directory at {storage}. "
            "Check ZOTERO_DIR, and ensure attachments are stored locally."
        )
    metadata = load_zotero_metadata(zotero_dir)
    by_key: dict[str, list[Path]] = defaultdict(list)
    for pdf_path in storage.glob("*/*.pdf"):
        by_key[pdf_path.parent.name].append(pdf_path)

    docs: list[Document] = []
    for key, paths in by_key.items():
        title, year = metadata.get(key, ("", ""))
        for i, pdf_path in enumerate(sorted(paths)):  # sorted -> deterministic ids
            doc_id = key if i == 0 else f"{key}__{pdf_path.stem}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title or pdf_path.stem,
                    year=year,
                    pdf_path=pdf_path,
                )
            )
    return docs


# === Extraction / chunking / images =========================================


def extract_pages(pdf_path: Path):
    """Yield per-page content from a PDF.

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
            try:
                text = page.get_text("text")
            except Exception:
                text = ""  # skip this page's text on failure rather than abort
            tables: list[str] = []
            try:
                for table in page.find_tables().tables:
                    markdown = table.to_markdown()
                    if markdown and markdown.strip():
                        tables.append(markdown.strip())
            except Exception:
                pass
            if text.strip() or tables:
                yield pno, text, tables
    finally:
        doc.close()  # never leak the file handle on a mid-iteration error


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

    Args:
        hits: Retrieved chunk dicts (each has ``pdf_path`` and ``page``).
        max_images: Maximum number of page images.

    Returns:
        PNG byte blobs, one per unique page, up to ``max_images``.
    """
    seen: list[tuple[str, int]] = []
    for hit in hits:
        key = (hit["pdf_path"], int(hit["page"]))
        if key not in seen:
            seen.append(key)
        if len(seen) >= max_images:
            break
    images: list[bytes] = []
    for pdf_path, page in seen:
        try:
            images.append(render_page_png(pdf_path, page))
        except Exception:
            continue
    return images


# === Dynamic LanceDB schema =================================================


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
        pdf_path: str
        text: str
        vector: Vector(dim)  # type: ignore[valid-type]

    return Chunk


# === Pipeline ===============================================================


class RAGPipeline:
    """End-to-end local RAG over a Zotero library.

    Args:
        zotero_dir: Path to the Zotero data directory.
        index_dir: Root directory for LanceDB, manifests, and the registry.
    """

    def __init__(self, zotero_dir: Path = ZOTERO_DIR, index_dir: Path = INDEX_DIR):
        self.zotero_dir = zotero_dir
        self.index_dir = index_dir
        self.lance_root = index_dir / "lancedb"
        self.lance_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.lance_root / "embedder_registry.json"
        self.db = lancedb.connect(self.lance_root)
        self._active_table = None  # cached after first prepare

    # --- naming helpers -----------------------------------------------------

    @staticmethod
    def _table_name(embedder_id_: str) -> str:
        """Return the LanceDB table name for an embedder id."""
        return f"chunks__{embedder_id_}"

    def _manifest_path(self, embedder_id_: str) -> Path:
        """Return the ingest manifest path for an embedder id."""
        return self.lance_root / f"ingested__{embedder_id_}.json"

    def _table_dir(self, embedder_id_: str) -> Path:
        """Return the on-disk table directory (for mtime fallbacks)."""
        return self.lance_root / f"{self._table_name(embedder_id_)}.lance"

    def _table_names(self) -> list[str]:
        """Return all LanceDB table names.

        Wraps ``list_tables()`` (``table_names()`` is deprecated). The single page
        returned covers this tool's small table count (at most
        ``MAX_EMBEDDER_TABLES`` plus a transient table during a switch).

        Returns:
            Table name strings.
        """
        result = self.db.list_tables()
        # lancedb >= 0.6 returns a paged object with a `.tables` attribute; older
        # versions (and possible future API churn) return a plain iterable. Fall
        # back to iterating directly so a shape change can't silently yield [],
        # which would make every lookup miss and trigger spurious re-ingests.
        if hasattr(result, "tables"):
            return list(result.tables)
        return [str(name) for name in result]  # type: ignore[union-attr]

    def _embedder_ids(self) -> set[str]:
        """Return the ids of all embedder tables currently on disk."""
        prefix = "chunks__"
        return {n[len(prefix):] for n in self._table_names() if n.startswith(prefix)}

    # --- registry -----------------------------------------------------------

    def _load_registry(self) -> dict[str, float]:
        """Load the embedder-id -> last_used (epoch) registry."""
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_registry(self, registry: dict[str, float]) -> None:
        """Persist the registry."""
        self.registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))

    def _reconcile_registry(self) -> dict[str, float]:
        """Make the registry agree with the tables actually on disk.

        Tables on disk are ground truth: adopt any orphan table (timestamp from
        its directory mtime) and drop any registry entry whose table is gone.

        Returns:
            The reconciled registry.
        """
        registry = self._load_registry()
        tables = self._embedder_ids()
        for tid in tables:
            if tid not in registry:
                try:
                    registry[tid] = self._table_dir(tid).stat().st_mtime
                except OSError:
                    registry[tid] = time.time()
        for tid in list(registry):
            if tid not in tables:
                del registry[tid]
        self._save_registry(registry)
        return registry

    def _touch(self, embedder_id_: str) -> None:
        """Mark an embedder as used now (once per run at prepare time)."""
        registry = self._load_registry()
        registry[embedder_id_] = time.time()
        self._save_registry(registry)

    # --- eviction / pruning -------------------------------------------------

    def _drop_embedder(self, embedder_id_: str) -> None:
        """Drop an embedder's table, manifest, and registry entry."""
        name = self._table_name(embedder_id_)
        if name in self._table_names():
            self.db.drop_table(name)
        manifest = self._manifest_path(embedder_id_)
        if manifest.exists():
            manifest.unlink()
        registry = self._load_registry()
        registry.pop(embedder_id_, None)
        self._save_registry(registry)
        print(f"Evicted embedder table: {embedder_id_}")

    def _enforce_cap(self) -> None:
        """Evict least-recently-used non-active tables down to the cap.

        Create-then-evict: callers build the new table first, so this trims any
        excess (len > cap) afterwards. The active embedder is never evicted.
        """
        registry = self._reconcile_registry()
        tables = self._embedder_ids()
        active = embedder_id()
        while len(tables) > MAX_EMBEDDER_TABLES:
            candidates = {t: registry.get(t, 0.0) for t in tables if t != active}
            if not candidates:
                break
            victim = min(candidates, key=lambda t: candidates[t])
            self._drop_embedder(victim)
            tables.discard(victim)
            registry.pop(victim, None)

    def keep_only(self, target_id: str | None = None, assume_yes: bool = False) -> None:
        """Drop every embedder table except one (interactive confirmation).

        Args:
            target_id: Embedder id to keep (defaults to the active embedder).
            assume_yes: Skip the confirmation prompt.
        """
        self._reconcile_registry()
        target = target_id or embedder_id()
        tables = self._embedder_ids()
        if target not in tables:
            print(f"No table for '{target}'. Available: {sorted(tables) or '(none)'}. Aborting.")
            return
        victims = sorted(t for t in tables if t != target)
        if not victims:
            print(f"Only '{target}' present; nothing to drop.")
            return
        print(f"Keeping: {target}\nWill DROP (irreversible; each needs re-ingest): {victims}")
        if not assume_yes:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return
        for tid in victims:
            self._drop_embedder(tid)
        print("Done.")

    def rebuild(self, assume_yes: bool = False) -> None:
        """Drop the active embedder's table + manifest, then re-ingest fresh.

        Rebuilds only the active embedder; other embedders' tables are left
        intact (for a full wipe, delete the cache directory). Interactive
        confirmation unless ``assume_yes``.

        Args:
            assume_yes: Skip the confirmation prompt.
        """
        active = embedder_id()
        name = self._table_name(active)
        has_table = name in self._table_names()
        suffix = "." if has_table else " (no existing table; just a fresh ingest)."
        print(f"Rebuild '{active}': drop its table and manifest, then re-ingest the library{suffix}")
        if not assume_yes:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return
        if has_table:
            self.db.drop_table(name)
        manifest = self._manifest_path(active)
        if manifest.exists():
            manifest.unlink()
        registry = self._load_registry()
        registry.pop(active, None)
        self._save_registry(registry)
        self._active_table = None  # invalidate cached handle
        self.ingest()

    # --- manifest -----------------------------------------------------------

    def _load_manifest(self, embedder_id_: str) -> set[str]:
        """Return the set of ingested doc ids for an embedder."""
        path = self._manifest_path(embedder_id_)
        if path.exists():
            return set(json.loads(path.read_text()))
        return set()

    def _save_manifest(self, embedder_id_: str, ids: set[str]) -> None:
        """Persist the ingested doc ids for an embedder."""
        self._manifest_path(embedder_id_).write_text(json.dumps(sorted(ids)))

    # --- index lifecycle ----------------------------------------------------

    def _verify_dim(self, embedder_id_: str) -> None:
        """Raise if a table's stored vector dim disagrees with the model's dim."""
        name = self._table_name(embedder_id_)
        if name not in self._table_names():
            return
        vector_field = self.db.open_table(name).schema.field("vector")
        stored = getattr(vector_field.type, "list_size", None)
        if stored is not None and stored != effective_dim():
            raise ValueError(
                f"Vector dim mismatch for '{embedder_id_}': table={stored}, "
                f"model={effective_dim()}. Delete the table or fix EMBEDDER."
            )

    def _open_active_table(self):
        """Prepare and open the active embedder's table.

        Reconciles the registry, ingests if the table is missing, marks the
        embedder used, verifies the dimension, and enforces the table cap.

        Returns:
            The active LanceDB table.
        """
        self._reconcile_registry()
        active = embedder_id()
        if self._table_name(active) not in self._table_names():
            print(f"No index for embedder '{active}' yet -- running ingest.")
            self.ingest()  # builds the table and runs its own touch/optimize/cap
        else:
            self._touch(active)
            self._verify_dim(active)
            self._enforce_cap()
        return self.db.open_table(self._table_name(active))

    def _prepare(self):
        """Open (once) and cache the active table for queries."""
        if self._active_table is None:
            self._active_table = self._open_active_table()
        return self._active_table

    def ingest(self) -> None:
        """Ingest documents not yet indexed for the active embedder. Safe to re-run."""
        active = embedder_id()
        name = self._table_name(active)
        ChunkModel = make_chunk_model(effective_dim())

        documents = discover_documents(self.zotero_dir)
        done = self._load_manifest(active)
        todo = [d for d in documents if d.doc_id not in done]
        print(f"[{active}] Found {len(documents)} PDFs; {len(todo)} new to ingest.")

        table = self.db.open_table(name) if name in self._table_names() else \
            self.db.create_table(name, schema=ChunkModel)

        # Crash-safe, idempotent ingest. Per-document upsert keyed on the
        # deterministic chunk id means re-running updates rather than appends
        # (table.add does no dedup, so the old append path silently doubled the
        # corpus on a retry). The manifest is persisted incrementally and in a
        # finally block, so an interrupt leaves a consistent state on resume.
        completed: set[str] = set(done)
        save_every = 25
        try:
            for i, doc in enumerate(tqdm(todo, desc="Ingesting", unit="doc"), start=1):
                try:
                    records = []
                    for page_no, page_text, tables in extract_pages(doc.pdf_path):
                        for idx, chunk_text in enumerate(chunk_page(page_text)):
                            records.append(self._record(doc, page_no, f"{page_no}:{idx}", chunk_text))
                        for tidx, table_md in enumerate(tables):
                            for sidx, piece in enumerate(_hard_split(table_md, CHUNK_CHARS)):
                                records.append(
                                    self._record(
                                        doc, page_no, f"{page_no}:t{tidx}:{sidx}", f"Table:\n{piece}"
                                    )
                                )
                    records = split_oversized_records(records)  # final size guard
                    if records:
                        vectors = embed_documents([r["text"] for r in records])
                        for record, vector in zip(records, vectors):
                            record["vector"] = vector
                        table.merge_insert("id") \
                            .when_matched_update_all() \
                            .when_not_matched_insert_all() \
                            .execute(records)
                    # Mark done on success, INCLUDING docs that open but yield no
                    # extractable text (so they aren't retried forever).
                    completed.add(doc.doc_id)
                except Exception as exc:
                    # One bad PDF must not abort the run; leave it UNMARKED so it
                    # is retried on the next run.
                    print(f"  warning: skipping {doc.pdf_path} ({exc!r}); will retry next run")
                    continue
                if i % save_every == 0:
                    self._save_manifest(active, completed)
        finally:
            # Persist whatever completed, even on Ctrl-C, so resume is consistent.
            self._save_manifest(active, completed)
            print(f"[{active}] Indexed {len(completed - done)} new documents.")

        # Reached only on NORMAL completion (skipped if the loop is interrupted),
        # so a crash mid-ingest never triggers eviction of the working table.
        table.optimize(cleanup_older_than=VERSION_RETENTION)
        self._touch(active)
        self._enforce_cap()

    @staticmethod
    def _record(doc: Document, page_no: int, suffix: str, text: str) -> dict:
        """Build a chunk record dict (without its vector).

        Args:
            doc: Source document.
            page_no: 1-based page number.
            suffix: Id suffix making the chunk unique within the document.
            text: Chunk text.

        Returns:
            A dict matching the dynamic Chunk schema minus ``vector``.
        """
        return {
            "id": f"{doc.doc_id}:{suffix}",
            "doc_id": doc.doc_id,
            "title": doc.title,
            "year": doc.year,
            "page": page_no,
            "pdf_path": str(doc.pdf_path),
            "text": text,
        }

    # --- retrieval / generation --------------------------------------------

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve and rerank the most relevant chunks for a question.

        Args:
            question: Search query (already rewritten, in chat mode).
            top_k: Number of chunks to return after reranking.

        Returns:
            The reranked top chunks.
        """
        table = self._prepare()
        query_vector = embed_query(question)
        candidates = table.search(query_vector).limit(RERANK_CANDIDATES).to_list()
        return rerank(question, candidates, top_k)

    def _build_context(self, hits: list[dict]) -> tuple[str, str]:
        """Format retrieved chunks into a context block and a source list.

        Args:
            hits: Retrieved chunk dicts.

        Returns:
            ``(context, sources)`` strings.
        """
        blocks, sources = [], []
        for n, hit in enumerate(hits, start=1):
            cite = hit["title"] + (f", {hit['year']}" if hit["year"] else "")
            blocks.append(f"[{n}] ({cite}, p.{hit['page']})\n{hit['text']}")
            sources.append(
                f"[{n}] {hit['title']}"
                + (f" ({hit['year']})" if hit["year"] else "")
                + f", p.{hit['page']}"
            )
        return "\n\n".join(blocks), "\n".join(sources)

    def _generate(self, question: str, hits: list[dict], history: list[dict] | None = None) -> str:
        """Generate a grounded answer via the configured provider.

        Args:
            question: The user's question for this turn (verbatim).
            hits: Retrieved chunks to ground the answer.
            history: Prior chat messages (role/content dicts), or None.

        Returns:
            The answer text (without the appended source list).
        """
        context, _ = self._build_context(hits)
        system = (
            "You are a precise research assistant. Answer ONLY using the provided "
            "sources. Cite every claim inline with [n] referring to the source "
            "numbers. If the sources do not contain the answer, say so plainly "
            "rather than guessing."
        )
        prompt = f"Sources:\n\n{context}\n\nQuestion: {question}"
        images = collect_page_images(hits) if MULTIMODAL else []
        if GEN_PROVIDER == "anthropic":
            return self._generate_anthropic(system, prompt, images, history)
        if GEN_PROVIDER == "cborg":
            return self._generate_cborg(system, prompt, images, history)
        return self._generate_ollama(system, prompt, images, history)

    def _generate_ollama(self, system: str, prompt: str, images: list[bytes], history) -> str:
        """Generate with a local Ollama chat model.

        Images are attached to the user message as raw PNG bytes (Ollama's native
        multimodal format), with a note appended to the prompt so the model knows
        to read them.

        Args:
            system: System prompt.
            prompt: User prompt text (sources followed by the question).
            images: Page images as raw PNG bytes, or empty for text-only.
            history: Prior chat messages (role/content dicts), or None.

        Returns:
            The model's reply text, stripped.
        """
        user_message: dict = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images
            user_message["content"] = (
                prompt + "\n\n(Page images of these sources are attached; use their "
                "figures and tables as needed.)"
            )
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append(user_message)
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=messages,
            think=False,
            options={"temperature": GEN_TEMPERATURE, "num_ctx": GEN_NUM_CTX},
        )
        content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        return (content or "").strip()

    def _generate_anthropic(self, system: str, prompt: str, images: list[bytes], history) -> str:
        """Generate with the Claude API via the Anthropic Messages API.

        Builds an ``anthropic.Anthropic`` client (which reads ``ANTHROPIC_API_KEY``
        from the environment) and issues a single Messages call against
        ``ANTHROPIC_MODEL``. Images are attached as base64 ``image`` content blocks
        before the prompt text.

        Args:
            system: System prompt (passed as the top-level ``system`` parameter).
            prompt: User prompt text, appended after any image blocks.
            images: Page images as raw PNG bytes, or empty for text-only.
            history: Prior chat messages (role/content dicts), or None.

        Returns:
            The concatenated text of the response's text blocks, stripped.
        """
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
        blocks: list[dict] = []
        for img in images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(img).decode("ascii"),
                    },
                }
            )
        blocks.append({"type": "text", "text": prompt})
        user_message: dict = {"role": "user", "content": blocks}
        messages: list[Any] = []
        if history:
            messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append(user_message)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _generate_cborg(self, system: str, prompt: str, images: list[bytes], history) -> str:
        """Generate via CBORG (LBNL), an OpenAI-compatible LiteLLM gateway.

        CBORG exposes the OpenAI chat-completions API (not the Anthropic Messages
        API), authenticated with a bearer token (``$CBORG_API_KEY``) against
        ``CBORG_BASE_URL``. The client is built with explicit ``api_key`` and
        ``base_url`` so it does not depend on whatever ``OPENAI_*`` variables
        happen to be exported in the shell, then delegated to
        :meth:`_generate_openai`.

        Args:
            system: System prompt.
            prompt: User prompt text (sources followed by the question).
            images: Page images as raw PNG bytes, or empty for text-only.
            history: Prior chat messages (role/content dicts), or None.

        Returns:
            The generated answer text, stripped.

        Raises:
            RuntimeError: If ``$CBORG_API_KEY`` is not set in the environment.
        """
        import openai

        token = os.environ.get("CBORG_API_KEY")
        if not token:
            raise RuntimeError(
                "GEN_PROVIDER='cborg' but $CBORG_API_KEY is not set in the environment."
            )
        if images:
            # CBORG's handling of image content is unverified; warn loudly rather
            # than fail silently if MULTIMODAL is routed through this provider.
            print(
                "WARNING: MULTIMODAL is on with GEN_PROVIDER='cborg', but CBORG's "
                "support for image content is UNVERIFIED -- this request may fail or "
                "silently ignore the attached page images.",
                file=sys.stderr,
            )
        client = openai.OpenAI(api_key=token, base_url=CBORG_BASE_URL)
        return self._generate_openai(
            client, CBORG_MODEL, CBORG_MAX_TOKENS, system, prompt, images, history
        )

    def _generate_openai(
        self,
        client,
        model: str,
        max_tokens: int,
        system: str,
        prompt: str,
        images: list[bytes],
        history,
    ) -> str:
        """Run one OpenAI chat-completions call (shared by OpenAI-compatible providers).

        Generic worker for any OpenAI-compatible endpoint: the caller supplies a
        configured client, model id, and token budget (e.g. :meth:`_generate_cborg`
        points it at the CBORG gateway). Images are attached as base64 data-URI
        ``image_url`` parts before the prompt text.

        Args:
            client: A configured ``openai.OpenAI`` client (api key and base URL
                already set by the caller).
            model: Model identifier or gateway alias to generate with.
            max_tokens: Maximum number of tokens to generate.
            system: System prompt, sent as a leading ``system`` message.
            prompt: User prompt text, appended after any image parts.
            images: Page images as raw PNG bytes, or empty for text-only.
            history: Prior chat messages (role/content dicts), or None.

        Returns:
            The response message content, stripped.
        """
        if images:
            # Images before the prompt text (matches _generate_anthropic and the
            # convention that a vision model attends to images, then the question).
            content: list[dict] = []
            for img in images:
                data = base64.standard_b64encode(img).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{data}"},
                    }
                )
            content.append({"type": "text", "text": prompt})
            user_message: dict = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": prompt}
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append(user_message)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=GEN_TEMPERATURE,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    def query(self, question: str, top_k: int = TOP_K) -> str:
        """Answer a single question (no memory).

        Args:
            question: Natural-language query.
            top_k: Chunks to retrieve as context.

        Returns:
            Answer with inline ``[n]`` citations and a numbered source list.
        """
        hits = self.retrieve(question, top_k)
        if not hits:
            return "No indexed content found. Run `ingest` first."
        answer = self._generate(question, hits)
        _, sources = self._build_context(hits)
        return f"{answer}\n\nSources:\n{sources}"

    def rewrite_query(self, history: list[dict], question: str) -> str:
        """Condense conversation + follow-up into a standalone search query.

        Args:
            history: Prior chat messages.
            question: The new follow-up message.

        Returns:
            A self-contained query string; falls back to ``question`` verbatim
            with no history or on failure.
        """
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-MAX_HISTORY_MESSAGES:])
        instruction = (
            "Rewrite the user's follow-up as a single standalone search query that "
            "captures the full topic, resolving pronouns and references from the "
            "conversation. Output ONLY the query, nothing else.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone query:"
        )
        try:
            resp = ollama.chat(
                model=REWRITE_MODEL,
                messages=[{"role": "user", "content": instruction}],
                think=False,
                options={"temperature": 0.0, "num_ctx": GEN_NUM_CTX},
            )
            text = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
            return (text or "").strip().strip('"') or question
        except Exception:
            return question

    def chat(self) -> None:
        """Run an interactive, history-aware chat loop over the library."""
        print("Zotero RAG chat. Type your question; '/exit' to quit.\n")
        history: list[dict] = []
        self._prepare()
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user in {"/exit", "/quit"}:
                break
            search_query = self.rewrite_query(history, user)
            hits = self.retrieve(search_query)
            if not hits:
                print("\nrag> No indexed content found. Run `ingest` first.\n")
                continue
            answer = self._generate(user, hits, history)
            _, sources = self._build_context(hits)
            print(f"\nrag> {answer}\n\n{sources}\n")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": answer})

    def stats(self) -> str:
        """Return a summary of the active config and all embedder tables."""
        registry = self._reconcile_registry()
        active = embedder_id()
        gen = {
            "anthropic": ANTHROPIC_MODEL,
            "cborg": CBORG_MODEL,
        }.get(GEN_PROVIDER, GEN_MODEL)
        lines = [
            f"Active embedder : {active}",
            f"Generation      : {GEN_PROVIDER} ({gen})",
            f"Reranker        : {RERANKERS[RERANKER].model} ({'on' if USE_RERANKER else 'off'})",
            f"Embedder tables (cap {MAX_EMBEDDER_TABLES}):",
        ]
        for tid in sorted(self._embedder_ids()):
            n = self.db.open_table(self._table_name(tid)).count_rows()
            docs = len(self._load_manifest(tid))
            ts = registry.get(tid)
            when = datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else "?"
            mark = "  <- active" if tid == active else ""
            lines.append(f"  {tid}: {docs} docs, {n} chunks, last_used {when}{mark}")
        return "\n".join(lines)


# === CLI ====================================================================


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Local Zotero RAG pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Index new PDFs for the active embedder.")
    q = sub.add_parser("query", help="Ask a single grounded question.")
    q.add_argument("question")
    q.add_argument("--top-k", type=int, default=TOP_K)
    sub.add_parser("chat", help="Interactive conversation with memory.")
    sub.add_parser("stats", help="Show config and index statistics.")
    ko = sub.add_parser("keep-only", help="Drop all embedder tables except one.")
    ko.add_argument("embedder_id", nargs="?", default=None, help="Defaults to the active embedder.")
    ko.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    rb = sub.add_parser("rebuild", help="Drop the active embedder's index and re-ingest from scratch.")
    rb.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    args = parser.parse_args(argv)
    pipeline = RAGPipeline()

    if args.command == "ingest":
        pipeline.ingest()
    elif args.command == "query":
        print(pipeline.query(args.question, args.top_k))
    elif args.command == "chat":
        pipeline.chat()
    elif args.command == "stats":
        print(pipeline.stats())
    elif args.command == "keep-only":
        pipeline.keep_only(args.embedder_id, args.yes)
    elif args.command == "rebuild":
        pipeline.rebuild(args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
