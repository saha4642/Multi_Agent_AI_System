# qa_mcp_ext_agent.py
from typing import Dict, Any

from agents import Agent, Runner


def build_mcp_ext_agent() -> Agent:
    return Agent(
        name="MCPExtenalHybridAgent",
        instructions=(
            "You receive an MCP `context` with: conversation, memory (mem: ids), "
            "documents (doc: ids), and external sources (ext: ids).\n"
            "Rules:\n"
            "1) Prefer DOCUMENTS for local/project specifics.\n"
            "2) Use EXTERNAL for background or when docs are insufficient.\n"
            "3) Use MEMORY to personalize or recall preferences/history.\n"
            "4) Cite as (doc: <chunk_id>), (mem: <mem_id>), (ext: <id>).\n"
            "5) Be concise and acknowledge uncertainty if sources conflict."
        ),
    )


async def answer_with_mcp_external(
    query: str,
    mcp_context: Dict[str, Any],
) -> str:
    agent = build_mcp_ext_agent()

    # Try passing MCP context natively to the Runner
    try:
        res = await Runner.run(
            agent,
            input=query,
            context=mcp_context.get("context"),
        )  # type: ignore
        return getattr(res, "final_output", str(res))

    except TypeError:
        # Fallback: inline MCP context into prompt
        ctx = mcp_context.get("context", {})
        prompt = (
            "Use this MCP context to answer. "
            "Cite (doc:..), (mem:..), (ext:..).\n\n"
            f"{ctx}\n\n"
            f"QUESTION: {query}"
        )

        res = await Runner.run(agent, input=prompt)
        return getattr(res, "final_output", str(res))
