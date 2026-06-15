"""Diversity-aware selection and query-time duplicate-copy canonicalization.

Two concerns, both applied AFTER cross-encoder scoring and BEFORE the final
top-k slice:

* :func:`select_diverse` -- greedy MMR that penalises redundancy only WITHIN the
  same ``doc_id`` (cross-document corroboration is preserved), plus a soft
  per-document cap.
* :func:`_canonicalize_by_title` -- collapses duplicate copies of the same work
  (same title, different DOI -- the duplicates Zotero misses), keeping the
  highest-scored copy. :func:`duplicate_title_groups` is the pure grouping
  primitive behind the ``stats --duplicates`` audit, sharing the same title key
  and year-window predicate.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np

from .config import MMR_LAMBDA, PER_DOC_CAP


def select_diverse(
    candidates: list[dict],
    scores: list[float],
    top_k: int,
    lambda_: float = MMR_LAMBDA,
    per_doc_cap: int = PER_DOC_CAP,
) -> list[dict]:
    """Select ``top_k`` chunks balancing relevance against within-doc redundancy.

    Greedy Maximal Marginal Relevance (MMR): at each step pick the candidate that
    maximises ``lambda_ * relevance - (1 - lambda_) * redundancy``, where
    redundancy is the largest cosine similarity to an already-selected chunk FROM
    THE SAME ``doc_id``. Cross-document similarity is deliberately NOT penalised,
    so two papers stating the same fact can both survive (corroboration).

    A soft per-document cap is applied in two passes: the first respects
    ``per_doc_cap``; if that cannot fill ``top_k``, a second pass admits overflow
    so ``min(top_k, len(candidates))`` chunks are always returned.

    Args:
        candidates: Candidate chunk dicts; each must carry an L2-normalised
            ``vector`` (LanceDB returns it from ``.to_list()``) and a ``doc_id``.
        scores: One relevance score per candidate (same order). Min-max
            normalised to [0, 1] internally for scale-stable mixing.
        top_k: Number of chunks to return.
        lambda_: Relevance weight in [0, 1]; ``1.0`` is pure relevance.
        per_doc_cap: Soft maximum chunks per ``doc_id`` in the first pass.

    Returns:
        Up to ``top_k`` selected chunk dicts, in selection order.
    """
    if not candidates:
        return []
    # Vectors are L2-normalised, so dot product == cosine similarity.
    vectors = np.asarray([c["vector"] for c in candidates], dtype=np.float32)
    raw = np.asarray(scores, dtype=np.float32)
    span = float(raw.max() - raw.min())
    rel = (raw - raw.min()) / span if span > 0 else np.zeros(len(raw), dtype=np.float32)

    selected: list[int] = []
    by_doc: dict[str, list[int]] = defaultdict(list)

    def best_candidate(respect_cap: bool) -> int | None:
        """Return the unselected index with the highest MMR score, or None."""
        best_idx, best_mmr = None, float("-inf")
        for i in range(len(candidates)):
            if i in selected:
                continue
            doc = candidates[i]["doc_id"]
            if respect_cap and len(by_doc[doc]) >= per_doc_cap:
                continue
            redundancy = max((float(vectors[i] @ vectors[j]) for j in by_doc[doc]), default=0.0)
            mmr = lambda_ * float(rel[i]) - (1.0 - lambda_) * redundancy
            if mmr > best_mmr:
                best_idx, best_mmr = i, mmr
        return best_idx

    # First pass respects the per-doc cap; second admits overflow to fill top_k.
    for respect_cap in (True, False):
        while len(selected) < top_k:
            idx = best_candidate(respect_cap)
            if idx is None:
                break
            selected.append(idx)
            by_doc[candidates[idx]["doc_id"]].append(idx)
        if len(selected) >= top_k:
            break
    return [candidates[i] for i in selected]


# === Title canonicalization (query-time duplicate-copy removal) ==============


def _normalize_title(title: str) -> str:
    """Normalize a title into a key for matching duplicate copies of a work.

    NFKD-normalizes, strips combining marks, casefolds, replaces every
    non-alphanumeric character with a space, then collapses whitespace and strips.
    Pure and deterministic; adds no new dependency (``unicodedata`` is stdlib).

    Args:
        title: Raw title (may be empty).

    Returns:
        The normalized matching key (possibly empty for a title with no
        alphanumerics).
    """
    decomposed = unicodedata.normalize("NFKD", title or "")
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = without_marks.casefold()
    spaced = "".join(ch if ch.isalnum() else " " for ch in folded)
    return " ".join(spaced.split())


def _title_year(year: str | None) -> int | None:
    """Parse a 4-digit year from a record's ``year`` field, or ``None`` if absent."""
    match = re.search(r"\d{4}", year or "")
    return int(match.group()) if match else None


def _years_compatible(
    year: int | None, anchor_year: int | None, year_window: int | None
) -> bool:
    """Return whether two years are close enough to be the same work.

    True when matching on title alone (``year_window`` is None) or when either year
    is missing; otherwise the years must differ by at most ``year_window``. A
    missing year therefore falls back to title-only matching for that candidate.

    Args:
        year: Candidate year, or None.
        anchor_year: Anchor year, or None.
        year_window: Max allowed absolute year difference, or None.

    Returns:
        Whether the candidate may join the anchor on the year test.
    """
    if year_window is None or year is None or anchor_year is None:
        return True
    return abs(year - anchor_year) <= year_window


def _canonicalize_by_title(
    candidates: list[dict],
    scores: list[float],
    year_window: int | None,
) -> tuple[list[dict], list[float]]:
    """Drop duplicate copies of the same work, keeping the highest-scored copy.

    Zotero leaves same-title items with different DOIs unmerged, so one work can
    appear under several ``doc_id``s (preprint vs published, arXiv vs journal) with
    near-identical content that :func:`select_diverse` deliberately does not
    penalize (cross-doc similarity is kept for corroboration). This collapses those
    at query time, keyed on the normalized title -- the field stable across
    versions.

    Candidates are processed in DESCENDING score order, so the first copy of a work
    seen (its highest-scored chunk) becomes that work's anchor and any lower-scored
    copy is the one dropped; processing in any other order could keep the worse
    copy. Each anchor records its kept ``doc_id`` and year. A candidate joins an
    existing anchor when their normalized titles are equal AND their years are
    compatible (see :func:`_years_compatible`); it is then KEPT only if it is the
    anchor's own ``doc_id`` (another chunk of the already-kept copy) and DROPPED
    otherwise (a different copy of the same work). A candidate matching no anchor
    establishes a new one and is kept -- so a generic title whose years fall outside
    the window yields two anchors = two distinct works, both kept. Different
    normalized titles are NEVER merged, so cross-paper corroboration is preserved.

    Honesty: a generic title shared by genuinely different works WILL merge if their
    years fall within ``year_window`` (rare; the window bounds it); a version whose
    title was edited beyond normalization will NOT merge (rare; titles are the
    stable field across versions); a missing year falls back to title-only matching
    for that candidate; and a title that normalizes to an empty key (no
    alphanumerics) is never anchored, so such candidates are always kept (no
    untitled-vs-untitled collapse).

    Args:
        candidates: Candidate chunk dicts (each with ``doc_id``, ``title``,
            ``year``).
        scores: One relevance score per candidate, aligned with ``candidates``.
        year_window: Max year difference for two same-title copies to count as one
            work; ``None`` matches on the normalized title alone.

    Returns:
        ``(kept_candidates, kept_scores)``, aligned and still in descending-score
        order.
    """
    # Per normalized title, a list of (kept doc_id, year) anchors -- more than one
    # only when years are incompatible (so a shared generic title can split into
    # distinct works).
    anchors: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    kept_candidates: list[dict] = []
    kept_scores: list[float] = []
    # Descending score so the higher-scored copy of a work anchors it and the
    # lower-scored copies are the ones dropped.
    for i in sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True):
        cand = candidates[i]
        key = _normalize_title(cand["title"])
        if not key:  # no title key to anchor on -> never merge, always keep
            kept_candidates.append(cand)
            kept_scores.append(scores[i])
            continue
        year = _title_year(cand.get("year", ""))
        anchor = next(
            (a for a in anchors[key] if _years_compatible(year, a[1], year_window)),
            None,
        )
        if anchor is not None:
            if cand["doc_id"] != anchor[0]:
                continue  # a different copy of the same work -> drop
        else:
            anchors[key].append((cand["doc_id"], year))  # new work -> new anchor
        kept_candidates.append(cand)
        kept_scores.append(scores[i])
    return kept_candidates, kept_scores


def duplicate_title_groups(
    rows: list[dict], year_window: int | None
) -> list[dict]:
    """Group rows by normalized title and cluster duplicate copies by year.

    Reuses :func:`_normalize_title` (the canonicalizer's title key) and
    :func:`_years_compatible` (its year-window predicate) so the audit mirrors
    what :func:`_canonicalize_by_title` would do, rather than reimplementing
    either. Only titles shared by more than one distinct ``doc_id`` are kept.

    Within each such title, the distinct copies are partitioned into
    sub-clusters by the SAME greedy year test the canonicalizer applies: a copy
    joins the first existing sub-cluster whose anchor (first member's) year is
    compatible (see :func:`_years_compatible`), else it starts a new one. One
    sub-cluster means the copies would collapse to a single work; more than one
    means they would be treated as distinct.

    NOTE: the live canonicalizer processes copies in descending query-score
    order, which this audit cannot know, so it clusters in a deterministic
    ``(year, doc_id)`` order instead. The verdict is therefore indicative --
    order can change greedy clustering when years chain across the window -- not
    a guarantee for any single query.

    Args:
        rows: Chunk rows, each with ``doc_id``, ``title`` and ``year`` keys.
        year_window: Max year difference for two same-title copies to count as
            one work; ``None`` matches on the normalized title alone.

    Returns:
        One dict per duplicated title, each with ``title`` (display title of the
        first occurrence), ``members`` (``(doc_id, year)`` pairs, year an int or
        ``None``) and ``clusters`` (the sub-clusters, each a list of members).
    """
    members_by_key: dict[str, dict[str, int | None]] = defaultdict(dict)
    display: dict[str, str] = {}
    for row in rows:
        key = _normalize_title(row["title"])
        if not key:  # untitled -> not title-matchable; mirrors the canonicalizer
            continue
        display.setdefault(key, row["title"])  # first occurrence's title
        # First year seen per doc_id (year is per-document, so stable).
        members_by_key[key].setdefault(row["doc_id"], _title_year(row.get("year", "")))

    groups: list[dict] = []
    for key, members_map in members_by_key.items():
        if len(members_map) < 2:
            continue
        # Deterministic order (present years ascending, missing last; then
        # doc_id) so the greedy clustering below is reproducible across runs.
        members = sorted(
            members_map.items(), key=lambda m: (m[1] is None, m[1] or 0, m[0])
        )
        clusters: list[list[tuple[str, int | None]]] = []
        for doc_id, year in members:
            anchor = next(
                (c for c in clusters if _years_compatible(year, c[0][1], year_window)),
                None,
            )
            if anchor is None:
                clusters.append([(doc_id, year)])
            else:
                anchor.append((doc_id, year))
        groups.append({"title": display[key], "members": members, "clusters": clusters})
    return groups
