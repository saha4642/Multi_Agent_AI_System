# multi_agent_collab.py

from __future__ import annotations

import asyncio
import json
import datetime
from typing import Any, Dict, List

from agents import Agent, Runner
from retriever import retrieve
from qa_agent import answer_with_context
from longterm_memory import recall_relevant


# =============================================================================
# Agent Builders
# =============================================================================

def build_researcher() -> Agent:
    return Agent(
        name="ResearcherAgent",
        instructions=(
            "You are ResearcherAgent. Use RAG context and produce detailed, factual insights. "
            "Return findings in JSON with keys: role, type, findings, citations."
        ),
    )


def build_summarizer() -> Agent:
    return Agent(
        name="SummarizerAgent",
        instructions=(
            "You are SummarizerAgent. You receive multiple research findings "
            "and condense them into a clear, readable summary. "
            "Return JSON with keys: role, type, summary, highlights."
        ),
    )


def build_evaluator() -> Agent:
    return Agent(
        name="EvaluatorAgent",
        instructions=(
            "You are EvaluatorAgent. Assess correctness and relevance of findings. "
            "Score accuracy (0-1), relevance (0-1), and provide brief comments. "
            "Return JSON with keys: role, type, accuracy, relevance, comments."
        ),
    )


def build_coordinator() -> Agent:
    return Agent(
        name="CoordinatorAgent",
        instructions=(
            "You are CoordinatorAgent. Integrate outputs from all agents "
            "into a final collaborative report in JSON with keys: "
            "role, type, final_answer, sources, quality_estimate."
        ),
    )


# =============================================================================
# Core Orchestration
# =============================================================================

async def run_multi_agent_collab(
    store,
    session,
    *,
    query: str,
    topk_docs: int = 5,
    topk_mem: int = 2,
):
    """
    Main routine for multi-agent collaborative querying and synthesis.
    """

    # -------------------------------------------------------------------------
    # Step 1: Retrieve RAG and memory context
    # -------------------------------------------------------------------------
    rag_matches = retrieve(query, store, top_k=topk_docs)
    mem_snippets = recall_relevant(query, top_k=topk_mem)

    rag_text = (
        await answer_with_context(query, rag_matches)
        if rag_matches
        else "No RAG context available."
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

    print("\n===============================================")
    print(" Hands-On: Test Multi-Agent Knowledge Queries ")
    print("===============================================\n")

    # -------------------------------------------------------------------------
    # Step 2: Researcher produces detailed findings
    # -------------------------------------------------------------------------
    researcher = build_researcher()

    r_prompt = (
        f"Query: {query}\n\n"
        "Use the context below to extract factual information.\n"
        "Return JSON with role, type, findings, and citations.\n\n"
        f"Context:\n{combined_context}"
    )

    r_out = await Runner.run(researcher, input=r_prompt, session=session)
    r_json = getattr(r_out, "final_output", str(r_out))
    print("[Researcher] Raw output:\n", r_json, "\n")

    # -------------------------------------------------------------------------
    # Step 3: Summarizer condenses it
    # -------------------------------------------------------------------------
    summarizer = build_summarizer()

    s_prompt = f"Summarize and condense these findings:\n{r_json}"
    s_out = await Runner.run(summarizer, input=s_prompt, session=session)
    s_json = getattr(s_out, "final_output", str(s_out))
    print("[Summarizer] Raw output:\n", s_json, "\n")

    # -------------------------------------------------------------------------
    # Step 4: Evaluator verifies accuracy/relevance
    # -------------------------------------------------------------------------
    evaluator = build_evaluator()

    e_prompt = f"Evaluate correctness and relevance of the following:\n{s_json}"
    e_out = await Runner.run(evaluator, input=e_prompt, session=session)
    e_json = getattr(e_out, "final_output", str(e_out))
    print("[Evaluator] Raw output:\n", e_json, "\n")

    # -------------------------------------------------------------------------
    # Step 5: Coordinator synthesizes final answer
    # -------------------------------------------------------------------------
    coordinator = build_coordinator()

    c_prompt = (
        "You are to integrate outputs from all agents below and produce a final "
        "collaborative answer.\n\n"
        f"Researcher findings:\n{r_json}\n\n"
        f"Summarizer output:\n{s_json}\n\n"
        f"Evaluator feedback:\n{e_json}\n\n"
        "Return a final JSON with keys: role, type, final_answer, sources, quality_estimate."
    )

    c_out = await Runner.run(coordinator, input=c_prompt, session=session)
    c_json = getattr(c_out, "final_output", str(c_out))

    print("[Coordinator] ✅ Final Collaborative Output:\n", c_json, "\n")

    return c_json
