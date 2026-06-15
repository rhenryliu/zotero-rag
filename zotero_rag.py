"""Entry point for the local Zotero RAG pipeline.

All application code lives in the :mod:`rag` package; this module is a thin
launcher so ``python zotero_rag.py <command>`` keeps working. See
``rag/__init__.py`` for the pipeline overview and ``rag/config.py`` for tuning.
"""

import sys

from rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
