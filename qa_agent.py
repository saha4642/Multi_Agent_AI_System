# qa_agent.py
from typing import List, Dict
from agents import Agent, Runner


def build_retriever_agent() -> Agent:
    return Agent(
        name="RetrieverQAAssistant",
        instructions=(
            "You answer questions using the provided CONTEXT snippets.\n"
            "Rules:\n"
            "- Use only the given context; if insufficient, say you are unsure.\n"
            "- Cite sources by chunk IDs in parentheses like (source: <id>).\n"
            "- Keep answers concise and accurate."
        ),
    )


async def answer_with_context(query: str, contexts: List[Dict]) -> str:
    """
    contexts: list of {'id','text','metadata','score'}
    Builds a prompt with top snippets then answers.
    """
    ctx_blocks = []
    for c in contexts:
        cid = c.get("id", "")
        txt = (c.get("text") or "").strip()
        ctx_blocks.append(f"[{cid}]\n{txt}")

    ctx_blob = "\n\n".join(ctx_blocks) if ctx_blocks else "(no context)"
    prompt = (
        "CONTEXT:\n"
        f"{ctx_blob}\n\n"
        f"QUESTION: {query}\n\n"
        "Write the best possible answer following the rules."
    )

    agent = build_retriever_agent()
    result = await Runner.run(agent, input=prompt)
    return getattr(result, "final_output", str(result))
