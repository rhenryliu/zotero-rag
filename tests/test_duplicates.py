import collections, numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zotero_rag import RAGPipeline, embed_query

p = RAGPipeline()
table = p._prepare()

# 1. Duplicate library entries: same title under >1 doc_id?
rows = table.to_pandas()[["doc_id", "title"]].drop_duplicates()
dupes = {t: n for t, n in collections.Counter(rows["title"]).items() if n > 1}
print("Titles with >1 doc_id (duplicate Zotero entries):")
for t, n in dupes.items():
    print(f"  {n}x  {t[:70]}")

# 2. The actual top candidates for the failing query: doc_id + pairwise cosine.
qv = embed_query("What are the best approaches for creating super resolution cosmological simulations")
cands = table.search(qv).limit(60).to_list()
print("\nTop candidates (id / doc_id / page / title):")
for c in cands[:14]:
    print(f"  {c['id']:<44} {c['doc_id'][:14]:<14} p{c['page']}-{c.get('page_end')}  {c['title'][:42]}")

V = np.asarray([c["vector"] for c in cands[:8]], dtype=np.float32)
print("\nPairwise cosine, top 8 (look for ~0.97+ = same text):")
print(np.round(V @ V.T, 3))