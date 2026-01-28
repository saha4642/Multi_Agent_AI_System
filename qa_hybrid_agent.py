# qa_hybrid_agent.py

from typing import List, Dict
from agents import Agent, Runner


def build_hybrid_agent() -> Agent:
    return Agent(
        name="HybridContextAssistant",
        instructions=(
            "You answer questions using BOTH MEMORY and DOCUMENT CONTEXT when available.\n"
            "Rules:\n"
            "- Prefer facts from DOCUMENT CONTEXT for specific details; use MEMORY to fill gaps or personalize.\n"
            "- If evidence conflicts, favor DOCUMENT CONTEXT and note uncertainty.\n"
            "- Cite sources inline by IDs: (doc: <chunk_id>) or (mem: <mem_id>).\n"
            "- If info is insufficient, say you're unsure.\n"
            "- Be concise and accurate."
        ),
    )


async def answer_hybrid(
    query: str,
    mem_items: List[Dict],
    doc_items: List[Dict],
) -> str:
    """
    mem_items: [{"id","text","score","type","session"}, ...]
    doc_items: [{"id","text","score","metadata"}, ...]
    """

    # -------------------------
    # Build MEMORY context
    # -------------------------
    mem_blocks = []
    for m in mem_items:
        mid = m.get("id", "")
        txt = (m.get("text") or "").strip()
        if txt:
            mem_blocks.append(f"[{mid}] {txt}")

    mem_blob = "\n".join(mem_blocks) if mem_blocks else "(none)"

    # -------------------------
    # Build DOCUMENT context
    # -------------------------
    doc_blocks = []
    for d in doc_items:
        did = d.get("id", "")
        txt = (
            d.get("text")
            or d.get("metadata", {}).get("text", "")
            or ""
        ).strip()
        if txt:
            doc_blocks.append(f"[{did}] {txt}")

    doc_blob = "\n".join(doc_blocks) if doc_blocks else "(none)"

    # -------------------------
    # Final prompt
    # -------------------------
    prompt = (
        f"QUESTION: {query}\n\n"
        f"MEMORY CONTEXT:\n{mem_blob}\n\n"
        f"DOCUMENT CONTEXT:\n{doc_blob}\n\n"
        "Write the best possible answer following the rules above."
    )

    agent = build_hybrid_agent()
    result = await Runner.run(agent, input=prompt)
    return getattr(result, "final_output", str(result))
