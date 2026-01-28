# retriever.py

from typing import List, Dict, Any
from vector_store import VectorStore


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    filter_doc_id: str | None = None
) -> List[Dict[str, Any]]:
    """
    Returns list of matches with 'id','score','metadata','text'.
    If filter_doc_id is provided, narrows to that document.
    Ensures 'text' is populated from metadata for backends that don't return it at top-level.
    """
    results = store.query(query, top_k=top_k * 3)  # overfetch slightly

    out: List[Dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata", {}) or {}

        if filter_doc_id and meta.get("doc_id") != filter_doc_id:
            continue

        # normalize text field for pinecone results
        if "text" not in r or not r.get("text"):
            r["text"] = meta.get("text", "")

        out.append(r)

        if len(out) >= top_k:
            break

    return out
