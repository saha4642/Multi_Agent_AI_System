# a2a_hybrid.py

from __future__ import annotations

import asyncio
import json
import datetime
from typing import Any, Dict, List

from agents import Agent, Runner
from longterm_memory import recall_relevant
from retriever import retrieve
from qa_agent import answer_with_context
from a2a_schema import A2AMessage  # your existing TaskSpec/Result dataclasses


# =============================================================================
# Builders
# =============================================================================

def build_planner() -> Agent:
    return Agent(
        name="PlannerAgent",
        instructions=(
            "You are a PlannerAgent. You define structured TaskSpecs for an Executor. "
            "Each TaskSpec must specify: task, required inputs, and expected structured output."
        ),
    )


def build_executor() -> Agent:
    return Agent(
        name="ExecutorAgent",
        instructions=(
            "You are an ExecutorAgent. You receive TaskSpecs in JSON form. "
            "Retrieve relevant context from documents (RAG) and memory, "
            "then respond with a structured Result JSON having keys: "
            "role, type, status, output, sources, and timestamp."
        ),
    )


# =============================================================================
# Core routine
# =============================================================================

async def run_a2a_hybrid(
    store,
    session,
    *,
    query: str,
    topk_docs: int = 5,
    topk_mem: int = 2,
) -> Dict[str, Any]:
    """
    Planner defines a TaskSpec that asks the Executor to integrate RAG + memory context.
    Executor fetches context, answers, and returns a structured Result.
    """

    # -------------------------------------------------------------------------
    # Step 1. Planner creates TaskSpec
    # -------------------------------------------------------------------------
    planner = build_planner()

    task_spec = {
        "role": "planner",
        "type": "TaskSpec",
        "content": {
            "task": "Integrate memory and RAG to answer: " + query,
            "params": {
                "rag_docs": topk_docs,
                "memory": topk_mem,
            },
        },
        "expected_output": "Structured summary integrating memory and RAG context.",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    planner_msg = json.dumps(task_spec, ensure_ascii=False, indent=2)
    print("\n[Planner → Executor] Sending TaskSpec:\n" + planner_msg + "\n")

    # -------------------------------------------------------------------------
    # Step 2. Executor processes it
    # -------------------------------------------------------------------------
    executor = build_executor()
    task = json.loads(planner_msg)
    user_query = query

    # Retrieve both contexts
    rag_matches = retrieve(user_query, store, top_k=topk_docs)
    mem_snippets = recall_relevant(user_query, top_k=topk_mem)

    rag_text = (
        await answer_with_context(user_query, rag_matches)
        if rag_matches
        else "No relevant RAG context."
    )

    mem_text = (
        "\n".join(mem_snippets)
        if mem_snippets
        else "No relevant memories."
    )

    combined_context = (
        f"RAG Context:\n{rag_text}\n\n"
        f"Memory Context:\n{mem_text}"
    )

    prompt = (
        "Given the user's query and the combined context below, "
        "generate a structured Result JSON with keys: "
        "role, type, status, output, sources, timestamp.\n\n"
        f"Query: {user_query}\n\n"
        f"Context:\n{combined_context}"
    )

    result = await Runner.run(
        executor,
        input=prompt,
        session=session,
    )

    raw_output = getattr(result, "final_output", str(result))

    # -------------------------------------------------------------------------
    # Step 3. Validate structured Result
    # -------------------------------------------------------------------------
    try:
        parsed = json.loads(raw_output)
        result_msg = A2AMessage(**parsed)
    except Exception as e:
        print("[Executor] ❌ Failed to parse Result JSON.")
        print("Raw output:\n", raw_output)
        print("Error:", e)
        return {
            "error": str(e),
            "raw": raw_output,
        }

    print("[Executor → Planner] ✅ Structured Result received:\n")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    return parsed
