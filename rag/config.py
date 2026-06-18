"""Central configuration for the Zotero RAG pipeline.

Almost all tuning is done by editing the constants in this module -- there are no
config files or env-based settings flags. The intended workflow is literally a
one-line edit (e.g. switching ``EMBEDDER`` or ``RERANKER``). Secrets are the only
exception: ``ANTHROPIC_API_KEY`` / ``CBORG_API_KEY`` come from the environment,
not from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

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
        sigmoid: Apply Sigmoid activation so scores fall in [0, 1]. Required for
            the relevance thresholds below to be meaningful -- raw logits are
            unbounded and model-specific, so ``min_score`` / ``soft_score`` are
            only set for sigmoid rerankers (a non-sigmoid reranker leaves them
            ``None`` -> no thresholding, current behavior preserved).
        min_score: Hard per-chunk floor in [0, 1]. Chunks scoring below this are
            dropped from the sources entirely; if NO chunk clears it the query is
            answered as "nothing sufficiently relevant". ``None`` disables it.
        soft_score: Marginal-vs-solid cut in [0, 1] (>= ``min_score``). Surviving
            chunks below this are kept but flagged "marginal relevance" in the
            citations and the model context; when EVERY survivor is marginal the
            answer carries a low-confidence caveat. ``None`` disables the tier.
    """

    model: str
    trust_remote_code: bool
    sigmoid: bool
    min_score: float | None = None
    soft_score: float | None = None


# Selectable rerankers. All three load through the same sentence-transformers
# CrossEncoder interface (per-model flags below). Reranking runs on MPS, NOT
# Ollama (which has no rerank endpoint). Switching needs no re-ingest.
RERANKERS: dict[str, RerankerSpec] = {
    # min_score/soft_score are on the SIGMOID [0,1] scale, calibrated against this
    # library's actual reranked nearest-neighbors (not contrived pairs, which look
    # falsely bimodal): off-domain queries top out low (pure biology ~0.026,
    # tangential semiconductor-vs-CMB-detector ~0.089) while a genuinely-soft ML
    # query reaches ~0.61. So the floor sits at 0.075 -- above clearly-irrelevant
    # queries (which then refuse locally with no generation call) but below loosely
    # adjacent ones (kept as flagged-marginal) -- and the marginal cut is 0.5, which
    # cleanly split the soft query (2 on-point chunks solid, the rest marginal).
    # Retune against your own library.
    "bge": RerankerSpec(
        "BAAI/bge-reranker-v2-m3", trust_remote_code=False, sigmoid=True,
        min_score=0.075, soft_score=0.50,
    ),
    # gte stays raw-logit (sigmoid=False) -> thresholds left None (unbounded scale).
    "gte": RerankerSpec("Alibaba-NLP/gte-reranker-modernbert-base", trust_remote_code=True, sigmoid=False),
    # Qwen3-Reranker is autoregressive -> higher per-pair latency than the BERT-based two.
    # Thresholds mirror bge's [0,1] bands (not separately calibrated yet).
    "qwen3": RerankerSpec(
        "Qwen/Qwen3-Reranker-0.6B", trust_remote_code=False, sigmoid=True,
        min_score=0.02, soft_score=0.50,
    ),
}
RERANKER = "bge"  # <-- one-line A/B switch: "bge" | "gte" | "qwen3"

USE_RERANKER = True
RERANK_DEVICE = "mps"  # falls back to CPU automatically if MPS load fails
RERANK_CANDIDATES = 32  # wide first-stage recall, before reranking
TOP_K = 8               # final chunks passed to the generator

# Diversity-aware final selection. When on (and a cross-encoder is available),
# ALL candidates are scored once and select_diverse trades relevance against
# within-document redundancy via greedy MMR, with a soft per-document cap. Off
# falls back to the plain rerank ordering.
SELECT_DIVERSE = True
MMR_LAMBDA = 0.7   # relevance vs within-doc redundancy (1.0 = pure relevance)
PER_DOC_CAP = 3    # soft max chunks per paper

# Query-time canonicalization of duplicate copies of the same work. Zotero does
# NOT flag same-title items with different DOIs as duplicates (the DOI mismatch
# vetoes the match), so preprint/published pairs and arXiv-vs-journal copies
# coexist as distinct doc_ids with near-identical content. select_diverse never
# penalizes cross-doc similarity (by design, to keep corroboration), so those
# copies surface as repeated sources. Keying on the title -- the field that
# survives versioning -- keeps one copy per work. Honest limits: a generic title
# shared by genuinely different works WILL merge if their years fall within the
# window (rare; the window bounds it), and a version whose title was edited beyond
# normalization will NOT merge (rare; titles are the stable field across versions).
# Switching needs NO re-ingest (operates on candidates already in the index).
TITLE_DEDUP = True
TITLE_DEDUP_YEAR_WINDOW = 1  # None = match on normalized title alone

# Generation backend.
GEN_PROVIDER = "cborg"  # "ollama" (local) | "anthropic" (Claude API) | "cborg" (LBNL gateway)
GEN_MODEL = "qwen3.6:27b"  # Ollama model. For MULTIMODAL must be vision + NON-MLX.
# Context window (tokens) for the local Ollama GENERATION call. Must fit the
# system prompt + TOP_K source chunks + chat history + question; Ollama defaults
# to a small 2048 window unless overridden, so this is set wide. Ignored by the
# anthropic/cborg providers (they use their own *_MAX_TOKENS budgets).
# GEN_NUM_CTX = 16384 # Needed for multimodal input on Ollama with the penalty of speed
GEN_NUM_CTX = 8192
GEN_TEMPERATURE = 0.2  # answer sampling; small but non-zero for fluent prose
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
# Context window (tokens) for the rewrite call. The rewriter only sees the chat
# history + a short instruction (NOT the source chunks), so it needs far less
# than GEN_NUM_CTX; sized just above worst-case history so its KV cache isn't
# oversized. The rewriter is a different model from GEN_MODEL, so a distinct
# num_ctx costs no extra model reload.
REWRITE_NUM_CTX = 4096
MAX_HISTORY_MESSAGES = 6  # chat turns (user+assistant) kept for rewrite & generation

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

# Figure-detection tunables (consumed by rag.figure_filter). Kept here with the
# rest of the pipeline's tunables so all configuration lives in one module.
FIGURE_MIN_AREA_FRAC = 0.03  # figure-vs-equation area gate (tunable)
FIGURE_PAD = 14.0            # region inflation in points
FIGURE_GRANULARITY = "line"  # "line" | "block"
