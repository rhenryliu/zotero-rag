"""Zotero library discovery: storage-folder walk + best-effort metadata.

``zotero.sqlite`` is opened read-only/immutable so the pipeline works while
Zotero is running. Metadata lookup is best-effort: any failure falls back to the
PDF filename.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
