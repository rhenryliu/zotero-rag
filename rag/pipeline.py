"""The end-to-end RAG pipeline over a Zotero library.

``RAGPipeline`` owns the LanceDB lifecycle (one table per embedder, registry,
eviction, crash-safe ingest) and the query path (retrieve -> rerank ->
canonicalize -> diversity-select -> generate). The stage helpers it composes live
in the sibling modules (:mod:`rag.embedding`, :mod:`rag.reranking`,
:mod:`rag.selection`, :mod:`rag.extraction`, :mod:`rag.chunking`,
:mod:`rag.generation`, ...).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import lancedb
import ollama
from tqdm import tqdm

from . import generation
from .chunking import _hard_split, chunk_document, split_oversized_records
from .config import (
    ANTHROPIC_MODEL,
    CBORG_MODEL,
    CHUNK_CHARS,
    GEN_MODEL,
    GEN_NUM_CTX,
    GEN_PROVIDER,
    INDEX_DIR,
    MAX_EMBEDDER_TABLES,
    MAX_HISTORY_MESSAGES,
    MULTIMODAL,
    RERANK_CANDIDATES,
    RERANKER,
    RERANKERS,
    REWRITE_MODEL,
    SELECT_DIVERSE,
    TITLE_DEDUP,
    TITLE_DEDUP_YEAR_WINDOW,
    TOP_K,
    USE_RERANKER,
    VERSION_RETENTION,
    ZOTERO_DIR,
)
from .embedding import effective_dim, embed_documents, embed_query, embedder_id
from .extraction import collect_page_images, extract_pages
from .library import Document, discover_documents
from .reranking import _cross_encoder_scores
from .schema import make_chunk_model
from .selection import _canonicalize_by_title, duplicate_title_groups, select_diverse


def _format_pages(page: int, page_end: int) -> str:
    """Format a page citation: ``p.N`` for one page, ``pp.N-M`` for a range.

    Args:
        page: First page of the chunk.
        page_end: Last page of the chunk (equals ``page`` for single-page chunks).

    Returns:
        A ``p.N`` or ``pp.N-M`` citation fragment.
    """
    return f"p.{page}" if page == page_end else f"pp.{page}-{page_end}"


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
                    # Prose is chunked across page breaks (chunk_document); tables
                    # are collected per page and NEVER routed through it, since
                    # cross-page packing would scramble a table's Markdown.
                    pages: list[tuple[int, str]] = []
                    table_records: list[dict] = []
                    for page_no, page_text, tables in extract_pages(doc.pdf_path):
                        pages.append((page_no, page_text))
                        for tidx, table_md in enumerate(tables):
                            for sidx, piece in enumerate(_hard_split(table_md, CHUNK_CHARS)):
                                table_records.append(
                                    self._record_range(
                                        doc, page_no, page_no,
                                        f"{page_no}:t{tidx}:{sidx}", f"Table:\n{piece}",
                                    )
                                )
                    records = [
                        self._record_range(doc, p_start, p_end, f"c{cidx}", chunk_text)
                        for cidx, (p_start, p_end, chunk_text) in enumerate(chunk_document(pages))
                    ]
                    records.extend(table_records)
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
    def _record_range(
        doc: Document, page_start: int, page_end: int, suffix: str, text: str
    ) -> dict:
        """Build a chunk record dict (without its vector).

        Args:
            doc: Source document.
            page_start: 1-based first page contributing to the chunk.
            page_end: 1-based last page contributing to the chunk; equals
                ``page_start`` for single-page chunks (e.g. tables).
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
            "page": page_start,
            "page_end": page_end,
            "pdf_path": str(doc.pdf_path),
            "text": text,
        }

    # --- retrieval / generation --------------------------------------------

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve and select the most relevant chunks for a question.

        Candidates are cross-encoder scored ONCE. With ``TITLE_DEDUP``,
        :func:`rag.selection._canonicalize_by_title` then collapses duplicate
        copies of the same work (same title, different DOI -- the duplicates Zotero
        misses) before selection, on every path. Finally, with ``SELECT_DIVERSE``
        and a working cross-encoder, the survivors are passed to
        :func:`rag.selection.select_diverse` (MMR + soft per-doc cap); otherwise
        they are ranked by score and sliced. A disabled or failing cross-encoder
        falls back to vector-search order (no dedup).

        Args:
            question: Search query (already rewritten, in chat mode).
            top_k: Number of chunks to return after selection.

        Returns:
            The selected top chunks.
        """
        table = self._prepare()
        query_vector = embed_query(question)
        candidates = table.search(query_vector).limit(RERANK_CANDIDATES).to_list()
        scores = _cross_encoder_scores(question, candidates)
        if scores is None:
            return candidates[:top_k]  # reranker disabled/unavailable -> vector order
        # Collapse duplicate copies of the same work BEFORE selection (and before
        # the non-diverse slice below), so dedup applies on every path. Needs the
        # scores, to keep the higher-scored copy of each work; with the reranker
        # unavailable (above) the vector order is returned undeduped.
        if TITLE_DEDUP:
            candidates, scores = _canonicalize_by_title(
                candidates, scores, TITLE_DEDUP_YEAR_WINDOW
            )
        if SELECT_DIVERSE:
            return select_diverse(candidates, scores, top_k)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [hit for hit, _ in ranked[:top_k]]

    def _build_context(self, hits: list[dict]) -> tuple[str, str]:
        """Format retrieved chunks into a context block and a source list.

        Args:
            hits: Retrieved chunk dicts.

        Returns:
            ``(context, sources)`` strings.
        """
        blocks, sources = [], []
        for n, hit in enumerate(hits, start=1):
            pages = _format_pages(hit["page"], hit["page_end"])
            cite = hit["title"] + (f", {hit['year']}" if hit["year"] else "")
            blocks.append(f"[{n}] ({cite}, {pages})\n{hit['text']}")
            sources.append(
                f"[{n}] {hit['title']}"
                + (f" ({hit['year']})" if hit["year"] else "")
                + f", {pages}"
            )
        return "\n\n".join(blocks), "\n".join(sources)

    def _generate(self, question: str, hits: list[dict], history: list[dict] | None = None) -> str:
        """Generate a grounded answer via the configured provider.

        Builds the system prompt, the sources-plus-question user prompt, and any
        page images (when ``MULTIMODAL``), then delegates the provider fan-out to
        :func:`rag.generation.generate`.

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
        return generation.generate(system, prompt, images, history)

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

    def _format_duplicate_audit(self, table) -> list[str]:
        """Render the duplicate-title audit lines for one embedder table.

        Reads only ``(doc_id, title, year)`` -- projecting columns avoids pulling
        vectors -- groups them via :func:`rag.selection.duplicate_title_groups`, and
        annotates each duplicated title with the verdict
        :func:`rag.selection._canonicalize_by_title` would reach: ``would merge``
        (one sub-cluster -> collapsed to one work), ``kept separate`` (every
        sub-cluster a singleton -> all distinct), or ``mixed`` (some sub-clusters
        merge, others stay separate; reported per sub-cluster). ``would merge``
        groups are the ones to eyeball -- distinct papers that share a title and
        close years would be silently collapsed -- so they sort first; ``kept
        separate`` confirms the year window is working. With ``TITLE_DEDUP`` off,
        the flat same-title list is shown instead, since no canonicalization runs
        at query time.

        Args:
            table: An open LanceDB table for one embedder.

        Returns:
            Indented report lines (header + entries) to append to the stats output.
        """
        rows = table.to_lance().to_table(columns=["doc_id", "title", "year"]).to_pylist()
        groups = duplicate_title_groups(rows, TITLE_DEDUP_YEAR_WINDOW)

        if not TITLE_DEDUP:
            lines = ["    Duplicate titles (TITLE_DEDUP off -- no query-time merge):"]
            if not groups:
                lines.append("      none")
                return lines
            for g in sorted(groups, key=lambda g: (-len(g["members"]), g["title"])):
                lines.append(f"      {len(g['members'])}x  {g['title']}")
            return lines

        window = (
            "title only" if TITLE_DEDUP_YEAR_WINDOW is None
            else f"TITLE_DEDUP_YEAR_WINDOW={TITLE_DEDUP_YEAR_WINDOW}"
        )
        lines = [f"    Duplicate titles (vs {window}):"]
        if not groups:
            lines.append("      none")
            return lines

        def label_of(group: dict) -> str:
            clusters = group["clusters"]
            if len(clusters) == 1:
                return "would merge"
            if all(len(c) == 1 for c in clusters):
                return "kept separate"
            return "mixed"

        def years_str(members: list[tuple[str, int | None]]) -> str:
            return ", ".join(str(y) if y is not None else "?" for _, y in members)

        labelled = [(label_of(g), g) for g in groups]
        # "would merge" first (the false-merge candidates to eyeball), then by count.
        labelled.sort(key=lambda e: (e[0] != "would merge", -len(e[1]["members"]), e[1]["title"]))
        for label, g in labelled:
            count = len(g["members"])
            if label == "mixed":
                lines.append(f"      {label:<13} {count}x  {g['title']}")
                for cluster in g["clusters"]:
                    verb = "merge" if len(cluster) > 1 else "separate"
                    lines.append(f"          - {len(cluster)}x {verb} ({years_str(cluster)})")
            else:
                lines.append(f"      {label:<13} {count}x  {g['title']}  ({years_str(g['members'])})")
        return lines

    def stats(self, show_duplicates: bool = False) -> str:
        """Return a summary of the active config and all embedder tables.

        Args:
            show_duplicates: When True, append a per-table duplicate-title audit
                (same normalized title under multiple ``doc_id``s), annotated with
                the verdict :func:`rag.selection._canonicalize_by_title` would reach
                for each group (see :meth:`_format_duplicate_audit`). The default
                stats output omits it.
        """
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
            table = self.db.open_table(self._table_name(tid))
            n = table.count_rows()
            docs = len(self._load_manifest(tid))
            ts = registry.get(tid)
            when = datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else "?"
            mark = "  <- active" if tid == active else ""
            lines.append(f"  {tid}: {docs} docs, {n} chunks, last_used {when}{mark}")
            if show_duplicates:
                lines.extend(self._format_duplicate_audit(table))
        return "\n".join(lines)
