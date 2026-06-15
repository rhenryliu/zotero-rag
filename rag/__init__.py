"""Local RAG over a Zotero library: retrieval + reranking + chat, on Ollama.

Everything runs locally by default. Generation can optionally be routed to the
Claude API (the only role that can be remote -- embedding and reranking always
run locally, since Claude is not an embedding or reranking model).

PIPELINE (stage by stage):
    1. Discovery   walk ~/Zotero/storage for PDFs; read titles/years from
                   zotero.sqlite (read-only), falling back to the filename.
                   (:mod:`rag.library`)
    2. Extraction  figure-aware text per page (PyMuPDF); tables -> Markdown too.
                   (:mod:`rag.extraction`, :mod:`rag.figure_filter`)
    3. Chunking    overlapping ~350-token chunks packed across page breaks;
                   each chunk cites a single page or a page range.
                   (:mod:`rag.chunking`)
    4. Embedding   chunks -> vectors via an Ollama embedding model (selectable).
                   (:mod:`rag.embedding`)
    5. Storage     LanceDB, ONE table per embedder (vectors of different models
                   are not comparable), keyed by model+dim. (:mod:`rag.schema`,
                   :mod:`rag.pipeline`)
    6. Retrieve    embed query, pull 24 nearest, cross-encoder rerank to top-k.
                   (:mod:`rag.reranking`, :mod:`rag.selection`)
    7. Generate    grounded answer with [n] citations via Ollama or Claude.
                   If MULTIMODAL, retrieved pages are rendered and attached so a
                   vision model can read figures/tables. (:mod:`rag.generation`)

MODES:
    query  one-shot question, no memory.
    chat   conversation with memory + history-aware query rewriting.

KEY CONFIG KNOBS (all in :mod:`rag.config`):
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
