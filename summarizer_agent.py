# summarizer_agent.py

from agents import Agent, Runner
from typing import List, Dict


def build_summarizer_agent() -> Agent:
    """Create a summarizer agent that condenses chat history."""
    return Agent(
        name="SummarizerAgent",
        instructions=(
            "You are a summarizer agent. Your goal is to read the entire conversation "
            "and produce a concise summary of the user's goals, important details, and agent responses. "
            "Avoid repetition and irrelevant greetings."
        ),
    )


async def summarize_history(history: List[Dict]) -> str:
    """Generate a summary from the chat history using the SummarizerAgent."""
    summarizer = build_summarizer_agent()

    text_blob = "\n".join(
        f"{m['role']}: {m['text']}"
        for m in history
    )

    prompt = (
        "Summarize the following conversation into 4–5 sentences. "
        "Capture user intentions and key details:\n\n"
        f"{text_blob}"
    )

    result = await Runner.run(
        summarizer,
        input=prompt
    )

    return getattr(result, "final_output", str(result))
