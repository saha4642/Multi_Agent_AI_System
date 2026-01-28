# interagent_chat.py

import asyncio
import json

from agents import Runner, SQLiteSession
from agent_config import build_agent as build_support_agent
from summarizer_agent import build_summarizer_agent


# You can switch roles or use any two compatible agents
def build_research_agent():
    agent = build_summarizer_agent()
    agent.name = "ResearchAgent"
    agent.instructions = (
        "You are a curious research assistant. "
        "Read messages from other agents and respond with clarifications, "
        "insights, or improvements. "
        "Keep your responses concise and helpful."
    )
    return agent


async def interagent_chat_demo(turns: int = 3) -> None:
    """Run a short two-agent conversation with independent sessions."""
    print("\n===============================================")
    print("Hands-On: Use AgentKit Sessions to Simulate Inter-Agent Chat")
    print("===============================================\n")

    # Create both agents
    support = build_support_agent()
    research = build_research_agent()

    # Independent memory sessions
    session_support = SQLiteSession(
        session_id="support_agent",
        db_path="support_session.sqlite",
    )
    session_research = SQLiteSession(
        session_id="research_agent",
        db_path="research_session.sqlite",
    )

    transcript = []

    msg = "Hello ResearchAgent, could you help summarize the latest retrieval workflow?"

    for t in range(1, turns + 1):
        print(f"\n[Turn {t}] SupportAgent → ResearchAgent")
        print(f"SupportAgent: {msg}")

        # ResearchAgent responds
        r_result = await Runner.run(
            research,
            input=msg,
            session=session_research,
        )
        r_reply = getattr(r_result, "final_output", str(r_result))

        print(f"ResearchAgent: {r_reply}")
        transcript.append({"speaker": "ResearchAgent", "text": r_reply})

        if t >= turns:
            break

        # SupportAgent replies back
        s_result = await Runner.run(
            support,
            input=r_reply,
            session=session_support,
        )
        s_reply = getattr(s_result, "final_output", str(s_result))

        print(f"\n[Turn {t}] ResearchAgent → SupportAgent")
        print(f"SupportAgent: {s_reply}")
        transcript.append({"speaker": "SupportAgent", "text": s_reply})

        # Prepare next message
        msg = s_reply

    # Save transcript
    with open("interagent_chat.jsonl", "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n[chat] Transcript saved to interagent_chat.jsonl\n")


if __name__ == "__main__":
    asyncio.run(interagent_chat_demo())
