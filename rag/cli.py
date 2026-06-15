"""Command-line entry point for the Zotero RAG pipeline."""

from __future__ import annotations

import argparse

from .config import TOP_K
from .pipeline import RAGPipeline


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
    st = sub.add_parser("stats", help="Show config and index statistics.")
    st.add_argument(
        "--duplicates",
        action="store_true",
        help="Also audit duplicate titles (same title, multiple doc_ids).",
    )
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
        print(pipeline.stats(args.duplicates))
    elif args.command == "keep-only":
        pipeline.keep_only(args.embedder_id, args.yes)
    elif args.command == "rebuild":
        pipeline.rebuild(args.yes)
    return 0
