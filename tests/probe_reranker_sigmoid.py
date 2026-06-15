"""Verify the active cross-encoder loads and scores sanely with Sigmoid on.

Confirms (1) loading with ``activation_fn=Sigmoid`` does not break the reranker
(the same path :func:`rag.reranking._get_reranker` exercises, including the
MPS->CPU fallback), (2) scores fall in [0, 1], and (3) clearly relevant pairs
outscore out-of-domain ones -- the separation the hard/soft thresholds in
``rag.config.RERANKERS`` rely on. Run:

    conda activate zotero-rag && python tests/probe_reranker_sigmoid.py
"""

from rag.config import RERANKER, RERANKERS
from rag.reranking import _cross_encoder_scores

# (question, [(label, passage), ...]); each question pairs one in-domain passage
# against one clearly off-domain one to expose the relevant-vs-irrelevant gap.
PROBES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "What is the matter power spectrum in cosmology?",
        [
            (
                "relevant",
                "The matter power spectrum P(k) characterises the amplitude of "
                "density fluctuations as a function of scale and is a central "
                "observable in large-scale structure cosmology.",
            ),
            (
                "off-domain",
                "Mitochondria are the powerhouse of the cell, generating ATP through "
                "oxidative phosphorylation across the inner membrane.",
            ),
        ],
    ),
    (
        "How does CRISPR-Cas9 perform targeted gene editing?",
        [
            (
                "relevant",
                "CRISPR-Cas9 uses a guide RNA to direct the Cas9 nuclease to a "
                "complementary DNA sequence, where it introduces a double-strand "
                "break for subsequent editing.",
            ),
            (
                "off-domain",
                "We constrain the dark energy equation of state using baryon acoustic "
                "oscillations measured in the galaxy two-point correlation function.",
            ),
        ],
    ),
]


def main() -> None:
    """Score each probe and print the per-passage relevance scores."""
    spec = RERANKERS[RERANKER]
    print(f"Reranker: {RERANKER} ({spec.model}), sigmoid={spec.sigmoid}")
    print(f"Thresholds: min_score={spec.min_score}, soft_score={spec.soft_score}")
    for question, passages in PROBES:
        hits = [{"text": text} for _, text in passages]
        scores = _cross_encoder_scores(question, hits)
        if scores is None:
            print("  reranker unavailable (USE_RERANKER off or load failed)")
            return
        print(f"\nQ: {question}")
        for (label, _), score in zip(passages, scores):
            print(f"  {label:>10}: {score:.4f}")


if __name__ == "__main__":
    main()
