# mcp_context.py

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from history_utils import format_ts
from longterm_memory import search_memory
from retriever import retrieve
from agents import SQLiteSession


def _recent_messages_preview(
    session: SQLiteSession,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """
    Pull a small window of recent turns from the AgentKit SQLiteSession
    to include in MCP context.

    We use the public API: session.get_messages(limit=?), which returns
    [{"role","content","created_at",...},...]

    If your SDK exposes a different shape, adapt here.
    """
    try:
        msgs = session.get_messages(limit=limit)  # type: ignore[attr-defined]
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for m in msgs:
        out.append(
            {
                "role": m.get("role", ""),
                "text": m.get("content", ""),
                "at": format_ts(m.get("created_at", time.time())),
            }
        )
    return out


def build_mcp_context(
    *,
    query: str,
    session: SQLiteSession,
    store,
    top_k_mem: int = 3,
    top_k_docs: int = 5,
    filter_doc_id: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns an MCP-style context dict you can pass to the agent runner.

    {
      "context": {
        "query": "...",
        "user_profile": {...},
        "memory": [{"id","text","score","type","session"}],
        "documents": [{"id","text","score","metadata":{...}}],
        "conversation": [{"role","text","at"}],
        "hints": {...}
      }
    }
    """

    # AgentKit long-term memory (your vectors)
    mem_hits = search_memory(query, top_k=top_k_mem)

    # RAG chunk retrieval
    doc_hits = retrieve(
        query,
        store,
        top_k=top_k_docs,
        filter_doc_id=filter_doc_id,
    )

    # Recent conversation (AgentKit SQLiteSession)
    conv = _recent_messages_preview(session, limit=6)

    ctx: Dict[str, Any] = {
        "context": {
            "query": query,
            "user_profile": user_profile or {},
            "memory": mem_hits,
            "documents": doc_hits,
            "conversation": conv,
            "hints": {
                "citations_format": (
                    "Use (doc: <chunk_id>) for document chunks and "
                    "(mem: <mem_id>) for memory."
                ),
                "conflict_resolution": (
                    "Prefer document evidence for specifics; "
                    "use memory to personalize."
                ),
            },
        }
    }

    return ctx
