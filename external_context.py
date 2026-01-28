# external_context.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents import SQLiteSession
from longterm_memory import search_memory
from retriever import retrieve
from mcp_external_tools import list_external_cache


def build_mcp_context_with_external(
    query: str,
    session: SQLiteSession,
    store,
    *,
    top_k_mem: int = 3,
    top_k_docs: int = 5,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds MCP-style context with an additional `external` field.

    Returns:
    {
        "context": {
            "query": "...",
            "user_profile": {...},
            "memory": [{id, text, score, type, session}],
            "documents": [{id, text, score, metadata}],
            "external": [{id, text, source, url, meta}],
            "conversation": [{role, text}],
            "hints": {...}
        }
    }
    """

    # --- Recent conversation (AgentKit SQLiteSession) ---
    try:
        conv = session.get_messages(limit=6)  # type: ignore[attr-defined]
        conversation = [
            {
                "role": m.get("role", ""),
                "text": m.get("content", ""),
            }
            for m in conv
        ]
    except Exception:
        conversation = []

    # --- Long-term memory (vector memory) ---
    mem_hits = search_memory(query, top_k=top_k_mem)

    # --- RAG document retrieval ---
    doc_hits = retrieve(
        query,
        store,
        top_k=top_k_docs,
        filter_doc_id=None,
    )

    # --- External cache (already fetched via MCP tools) ---
    ext = list_external_cache()

    return {
        "context": {
            "query": query,
            "user_profile": user_profile or {},
            "memory": mem_hits,
            "documents": doc_hits,
            "external": ext,
            "conversation": conversation,
            "hints": {
                "citations_format": (
                    "Use (doc: <chunk_id>) for documents, "
                    "(mem: <mem_id>) for memory, and "
                    "(ext: <id>) for external sources."
                ),
                "conflict_resolution": (
                    "Prefer documents for project specifics, "
                    "external sources for background knowledge, "
                    "and memory for personalization."
                ),
            },
        }
    }
