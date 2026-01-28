# a2a_share_demo.py
# ------------------------------------------------------------
# Hands-On: Share Results via A2A Protocol Schema
# ------------------------------------------------------------
# This demo shows two agents exchanging structured A2A messages.
# 1) PlannerAgent sends a TaskSpec
# 2) ExecutorAgent performs it
# 3) Planner validates the structured Result response
# ------------------------------------------------------------

import asyncio
import json

from agents import Runner, SQLiteSession
from agent_config import build_agent
from summarizer_agent import build_summarizer_agent
from a2a_protocol import A2AMessage


def build_executor_agent():
    agent = build_summarizer_agent()
    agent.name = "ExecutorAgent"
    agent.instructions = (
        "You are an ExecutorAgent. You receive structured task specifications "
        "in JSON and respond with structured results (status, output, source)."
    )
    return agent


async def a2a_share_demo():
    print("\n===============================================")
    print("Hands-On: Share Results via A2A Protocol Schema")
    print("===============================================\n")

    # Create agents
    planner = build_agent()
    planner.name = "PlannerAgent"

    executor = build_executor_agent()

    # Independent sessions
    planner_session = SQLiteSession(
        session_id="planner_session",
        db_path="planner.sqlite",
    )
    executor_session = SQLiteSession(
        session_id="executor_session",
        db_path="executor.sqlite",
    )

    # 1) Planner creates a structured task
    task_content = {
        "task": "Summarize local retrieval workflow",
        "params": {"context_doc": "retrieval.txt"},
        "expected_output": "Short technical summary of retrieval process",
    }

    task_msg = A2AMessage.create(
        role="planner",
        type_="TaskSpec",
        content=task_content,
    )

    print("[Planner → Executor] Sending structured TaskSpec:")
    print(task_msg.to_json())

    # 2) Executor receives it, processes, and replies with structured result
    result_prompt = f"""
You have received the following TaskSpec JSON:

{task_msg.to_json()}

Please produce a valid A2A Result JSON with fields:
- role: "executor"
- type: "Result"
- status: "success"
- output: a 1-sentence summary of the requested task
- source: "ExecutorAgent"

Return only the JSON, no extra text.
"""

    exec_result = await Runner.run(
        executor,
        input=result_prompt,
        session=executor_session,
    )
    result_text = getattr(exec_result, "final_output", str(exec_result))

    # 3) Planner validates structured result
    try:
        result_msg = A2AMessage.from_json(result_text)
        print("\n[Executor → Planner] Received structured Result:")
        print(result_msg.to_json())
        print("\n[Planner] ✅ Result validated and parsed successfully.")
    except Exception as e:
        print("\n[Planner] ❌ Failed to parse Result JSON.")
        print("Raw output:\n", result_text)
        print("Error:", e)


if __name__ == "__main__":
    asyncio.run(a2a_share_demo())
