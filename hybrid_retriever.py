# hybrid_retriever.py

from typing import List, Dict, Any, Optional

from vector_store import VectorStore
from retriever import retrieve
from longterm_memory import search_memory


def build_hybrid_context(
    query: str,
    store: VectorStore,
    top_k_docs: int = 5,
    top_k_mem: int = 3,
    filter_doc_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
    {
        "memory": [{"id","text","score","type","session"}, ...],
        "docs":   [{"id","text","score","metadata"}, ...]
    }
    """

    mem_items = search_memory(query, top_k=top_k_mem)

    doc_items = retrieve(
        query,
        store,
        top_k=top_k_docs,
        filter_doc_id=filter_doc_id,
    )

    return {
        "memory": mem_items,
        "docs": doc_items,
    }
