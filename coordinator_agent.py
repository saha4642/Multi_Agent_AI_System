# coordinator_agent.py
# ------------------------------------------------------------
# Fan-in answers from:
#   - RAG (retriever + QA)
#   - Long-term memory recall
#   - (Optional) External context via MCP, if available
# Then synthesize a single, cited answer with a confidence.
# ------------------------------------------------------------

from __future__ import annotations

import asyncio
import re
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner, SQLiteSession

# ------------------------------------------------------------
# Optional imports (graceful fallbacks)
# ------------------------------------------------------------
try:
    # Your existing retrieval & QA helpers from prior demos
    from retriever import retrieve  # (query, store, top_k, filter_doc_id=None) -> List[dict]
except Exception:
    retrieve = None  # type: ignore

try:
    from qa_agent import answer_with_context  # async (query, matches) -> str
except Exception:
    async def answer_with_context(query: str, matches: List[Dict[str, Any]]) -> str:
        # Minimal fallback: stitch top snippets
        ctx = []
        for m in matches[:3]:
            meta = m.get("metadata", {})
            src = meta.get("doc_id") or meta.get("source") or meta.get("name") or "doc"
            snippet = (m.get("text") or "")[:240]
            ctx.append(f"- [{src}] {snippet}")
        if not ctx:
            return "I couldn't find relevant context to answer."
        return "Here's a synthesized answer from available context:\n" + "\n".join(ctx)

try:
    # Long-term memory helpers already in your repo
    from longterm_memory import recall_relevant
except Exception:
    def recall_relevant(query: str, top_k: int = 2) -> List[str]:  # type: ignore
        return []

# ------------------------------------------------------------
# External-aware MCP builder if present
# ------------------------------------------------------------
_HAS_EXTERNAL = False
_build_ctx = None

try:
    from external_context import build_mcp_context_with_external as _build_ctx  # type: ignore
    _HAS_EXTERNAL = True
except Exception:
    try:
        from mcp_context import build_mcp_context as _build_ctx  # type: ignore
    except Exception:
        _build_ctx = None  # type: ignore

# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------
@dataclass
class PartialAnswer:
    source: str            # "rag" | "memory" | "external"
    text: str
    citations: List[str]
    confidence: float      # 0..1


def _extract_doc_citations(text: str) -> List[str]:
    """
    Pull inline (doc: X) / (mem: Y) / (ext: Z) citations if your QA format adds them.
    Fallback: look for doc:XYZ#chunk patterns anywhere.
    """
    cites: List[str] = []
    cites += re.findall(r"\((doc:[^)]+)\)", text or "")
    cites += re.findall(r"\((mem:[^)]+)\)", text or "")
    cites += re.findall(r"\((ext:[^)]+)\)", text or "")
    # also capture bare doc:ID#chunk
    cites += re.findall(r"\b(doc:[\w\-]+#\d{1,4})\b", text or "")
    # de-dupe, keep order
    seen = set()
    out = []
    for c in cites:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

# ------------------------------------------------------------
# Fan-in producers
# ------------------------------------------------------------
async def _rag_producer(
    query: str,
    store,
    top_k_docs: int,
    filter_doc_id: Optional[str],
) -> PartialAnswer:
    if retrieve is None:
        return PartialAnswer("rag", "RAG unavailable (retriever missing).", [], 0.1)

    matches = retrieve(query, store, top_k=top_k_docs, filter_doc_id=filter_doc_id)  # type: ignore
    if not matches:
        return PartialAnswer("rag", "No relevant documents found.", [], 0.2)

    answer = await answer_with_context(query, matches)
    cites = _extract_doc_citations(answer)

    # naive confidence: top score if present
    top_score = float(matches[0].get("score", 0.0))
    conf = max(0.1, min(1.0, 0.5 + 0.5 * top_score))
    return PartialAnswer("rag", answer, cites, conf)


async def _memory_producer(query: str, top_k_mem: int) -> PartialAnswer:
    mems = recall_relevant(query, top_k=top_k_mem) if top_k_mem > 0 else []
    if not mems:
        return PartialAnswer("memory", "No relevant long-term memories.", [], 0.2)

    joined = "\n".join(f"- {m}" for m in mems)
    # add synthetic mem citations
    cites = [f"mem:{i:06d}" for i, _ in enumerate(mems)]
    conf = 0.5 if len(mems) >= 2 else 0.35
    return PartialAnswer("memory", f"Relevant memories:\n{joined}", cites, conf)


async def _external_producer(
    query: str,
    session: SQLiteSession,
    store,
    top_k_docs: int,
    top_k_mem: int,
) -> PartialAnswer:
    if not _HAS_EXTERNAL or _build_ctx is None:
        return PartialAnswer("external", "External context disabled or unavailable.", [], 0.1)

    try:
        ctx = _build_ctx(
            query=query,
            session=session,
            store=store,
            top_k_docs=top_k_docs,
            top_k_mem=top_k_mem,
            include_external=True,  # only respected by external-aware builder
        )
        ext_items = (ctx.get("context", {}) or {}).get("external", [])  # type: ignore
        if not ext_items:
            return PartialAnswer("external", "No external hits.", [], 0.2)

        # build a dense digest
        lines = []
        cites = []
        for it in ext_items[:3]:
            src = it.get("source") or it.get("url") or "external"
            snippet = (it.get("text") or "")[:240]
            lines.append(f"- [{src}] {snippet}")
            cites.append(f"ext:{src}")

        text = "External context:\n" + "\n".join(lines)
        # de-dupe citations, keep order
        cites = list(dict.fromkeys(cites))
        return PartialAnswer("external", text, cites, 0.55)

    except Exception as e:
        return PartialAnswer("external", f"External context error: {e}", [], 0.15)

# ------------------------------------------------------------
# Coordinator synthesis
# ------------------------------------------------------------
COORDINATOR_SYS = (
    "You are a Coordinator Agent. You receive multiple partial answers "
    "from different sources (RAG, Memory, External).\n"
    "SYNTHESIZE a single, concise answer (5–10 lines) that:\n"
    "- Unifies overlapping facts\n"
    "- Clearly states caveats or disagreements\n"
    "- Appends a compact 'Sources:' line with unique citations like "
    "doc:..., mem:..., ext:...\n"
    "Return plain text (no markdown tables)."
)

def _build_coordinator() -> Agent:
    return Agent(name="CoordinatorAgent", instructions=COORDINATOR_SYS)


def _mk_coord_prompt(query: str, parts: List[PartialAnswer]) -> str:
    bundle = {
        "question": query,
        "partials": [asdict(p) for p in parts],
    }
    return "SYNTHESIZE THIS JSON INPUT:\n" + json.dumps(
        bundle, ensure_ascii=False, indent=2
    )

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
async def coordinator_answer(
    *,
    query: str,
    session: SQLiteSession,
    store: Any,
    top_k_docs: int = 5,
    top_k_mem: int = 2,
    filter_doc_id: Optional[str] = None,
    include_external: bool = True,
) -> Dict[str, Any]:
    """
    Fan-in RAG + Memory + (optional) External, then synthesize.

    Returns:
      {
        "partials": [PartialAnswer... as dict],
        "final": "text answer"
      }
    """
    tasks = [
        _rag_producer(query, store, top_k_docs, filter_doc_id),
        _memory_producer(query, top_k_mem),
    ]
    if include_external:
        tasks.append(_external_producer(query, session, store, top_k_docs, top_k_mem))

    parts: List[PartialAnswer] = list(await asyncio.gather(*tasks))

    coord = _build_coordinator()
    prompt = _mk_coord_prompt(query, parts)
    res = await Runner.run(coord, input=prompt, session=session)
    final_text = getattr(res, "final_output", str(res))

    return {
        "partials": [asdict(p) for p in parts],
        "final": final_text,
    }
