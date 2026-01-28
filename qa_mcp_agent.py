# qa_mcp_agent.py

from typing import Dict, Any, List
from agents import Agent, Runner


def build_mcp_agent() -> Agent:
    return Agent(
        name="MCPHybridAgent",
        instructions=(
            "You receive a structured MCP `context` field containing:\n"
            "- conversation: recent messages\n"
            "- memory: long-term notes/summaries (mem: ids)\n"
            "- documents: retrieved chunks (doc: ids)\n\n"
            "Rules:\n"
            "1) Prefer facts from DOCUMENTS for specifics; use MEMORY to personalize/fill gaps.\n"
            "2) Cite as (doc: <chunk_id>) and (mem: <mem_id>) where applicable.\n"
            "3) If evidence is insufficient, say you're unsure.\n"
            "4) Be concise, accurate, and actionable."
        ),
    )


async def answer_with_mcp_context(
    query: str,
    mcp_context: Dict[str, Any],
) -> str:
    """
    Expects mcp_context like: {"context": {...}} built by mcp_context.build_mcp_context().

    The Runner here is assumed to accept a `context` kwarg.
    If your Runner doesn't support a context kwarg, fall back to inlining
    the context into the prompt.
    """
    agent = build_mcp_agent()

    # Path A: Runner supports passing context
    try:
        res = await Runner.run(
            agent,
            input=query,
            context=mcp_context.get("context"),
        )  # type: ignore
        return getattr(res, "final_output", str(res))
    except TypeError:
        # Path B: Inline context into the prompt (fallback)
        ctx = mcp_context.get("context", {})
        preface = (
            "You are given the following MCP context JSON. Use it to answer.\n\n"
            f"{ctx}\n\n"
            f"QUESTION: {query}\n"
            "Remember to cite as (doc: <chunk_id>) and (mem: <mem_id>)."
        )
        res = await Runner.run(agent, input=preface)
        return getattr(res, "final_output", str(res))
