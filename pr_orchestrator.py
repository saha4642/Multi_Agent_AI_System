# pr_orchestrator.py

from __future__ import annotations
from typing import Any, Dict, List, Optional

from pr_protocol import parse_json_message, to_json
from pr_agents import run_planner, run_researcher, run_finalizer

# Try to use the external-aware context builder; fall back to a basic one.
# Both builders are expected to return a dict with a "context" sub-dict that
# includes keys like "documents", "memory", "external", etc.
try:
    from external_context import build_mcp_context_with_external as _build_ctx  # type: ignore
    HAS_EXTERNAL = True
except Exception:
    from mcp_context import build_mcp_context as _build_ctx  # type: ignore
    HAS_EXTERNAL = False

# Toggle verbose prints for raw/parsed JSON while debugging
DEBUG: bool = False


def _mk_context(
    *,
    query: str,
    session,
    store,
    top_k_docs: int,
    top_k_mem: int,
    filter_doc_id: Optional[str],
    include_external: bool,
) -> Dict[str, Any]:
    """
    Build an MCP-style context payload. We pass through optional knobs like
    'filter_doc_id' and 'include_external' if the builder supports them.
    """
    kwargs: Dict[str, Any] = dict(
        query=query,
        session=session,
        store=store,
        top_k_docs=top_k_docs,
        top_k_mem=top_k_mem,
        user_profile="demo: Planner-Researcher JSON",
    )

    # Best-effort: only include controls if the builder likely supports them.
    if filter_doc_id:
        kwargs["filter_doc_id"] = filter_doc_id
    if HAS_EXTERNAL:
        kwargs["include_external"] = include_external

    try:
        return _build_ctx(**kwargs)
    except TypeError:
        # Older builders may not accept these params; retry with the basics.
        fallback = dict(
            query=query,
            session=session,
            store=store,
            top_k_docs=top_k_docs,
            top_k_mem=top_k_mem,
        )
        return _build_ctx(**fallback)


async def planner_researcher_dialogue(
    *,
    question: str,
    session,
    store,
    top_k_docs: int = 5,
    top_k_mem: int = 3,
    max_turns: int = 2,
    # New: force-grounding controls for demos
    filter_doc_id: Optional[str] = None,  # e.g., "Retrieval"
    include_external: bool = True,         # set False to avoid ext. citations
) -> Dict[str, Any]:
    """
    Orchestrates a JSON-only dialogue:
      1) Planner emits a task spec (JSON)
      2) Researcher returns findings (JSON), optionally multiple rounds
      3) Planner finalizes answer (JSON)

    Returns:
      {
        "planner_spec": {...},
        "research_rounds": [{"json": {...}}, ...],
        "final": {...}
      }
    """
    out: Dict[str, Any] = {
        "planner_spec": {},
        "research_rounds": [],
        "final": {},
    }

    # 1) Planner: create task spec
    p_raw = await run_planner(question)
    if DEBUG:
        print("\n[DEBUG] Raw Planner Output:\n", p_raw)

    p_json = parse_json_message(p_raw)
    if DEBUG:
        print("[DEBUG] Parsed Planner JSON:\n", p_json)

    if not p_json or not isinstance(p_json, dict) or p_json.get("type") != "task_spec":
        print("[WARN] Planner output did not parse as valid 'task_spec' JSON.")
    out["planner_spec"] = p_json

    # 2) Build the MCP context with our grounding knobs
    ctx = _mk_context(
        query=question,
        session=session,
        store=store,
        top_k_docs=top_k_docs,
        top_k_mem=top_k_mem,
        filter_doc_id=filter_doc_id,
        include_external=include_external,
    )
    ctx_dict = ctx.get("context", {})

    # First researcher prompt is the planner JSON itself (or a minimal stub)
    r_prompt = to_json(p_json) if p_json else {
        "role": "planner",
        "type": "task_spec",
        "task": question,
    }

    # 3) Research rounds (bounded)
    for turn in range(max_turns):
        r_raw = await run_researcher(r_prompt, context=ctx_dict)
        if DEBUG:
            print(f"\n[DEBUG] Raw Researcher Output (round {turn + 1}):\n", r_raw)

        r_json = parse_json_message(r_raw)
        if DEBUG:
            print(f"[DEBUG] Parsed Researcher JSON (round {turn + 1}):\n", r_json)

        if not r_json or not isinstance(r_json, dict) or r_json.get("type") != "findings":
            print(f"[WARN] Researcher round {turn + 1} did not parse as valid 'findings' JSON.")

        # Warn if the researcher didn't cite any docs, especially when asked for doc grounding
        cites = (r_json or {}).get("citations", []) or []
        if filter_doc_id and not any(str(c).startswith("doc:") for c in cites):
            print(
                "[WARN] No document citations — check /rag_docs, your doc filter "
                f"('{filter_doc_id}'), and query terms to ensure they match the doc text."
            )

        out["research_rounds"].append({"json": r_json})

        # Stop if researcher does not request refinement or if there are no gaps
        next_action = (r_json or {}).get("next_action", "").lower()
        gaps: List[str] = (r_json or {}).get("gaps", []) or []
        if next_action != "plan_refine" or not gaps:
            break

        # Simple refinement message for the next researcher round
        refine_msg = {
            "role": "planner",
            "type": "refine",
            "address_gaps": gaps,
        }
        r_prompt = to_json(refine_msg)

    # 4) Finalization by Planner (final JSON answer)
    final_prompt = to_json({
        "role": "planner",
        "type": "finalize",
        "question": question,
        "spec": out["planner_spec"],
        "evidence": [rr["json"] for rr in out["research_rounds"]],
    })

    f_raw = await run_finalizer(final_prompt, context=ctx_dict)
    if DEBUG:
        print("\n[DEBUG] Raw Finalizer Output:\n", f_raw)

    f_json = parse_json_message(f_raw)
    if DEBUG:
        print("[DEBUG] Parsed Final JSON:\n", f_json)

    if not f_json or not isinstance(f_json, dict) or f_json.get("type") != "final_answer":
        print("[WARN] Finalizer output did not parse as valid 'final_answer' JSON.")

    # Warn on missing doc citations in the final if we attempted doc grounding
    final_cites = (f_json or {}).get("citations", []) or []
    if filter_doc_id and not any(str(c).startswith("doc:") for c in final_cites):
        print(
            "[WARN] Final answer has no document citations — ensure the Researcher "
            "found doc evidence. Consider setting top_k_mem=0, clearing external cache, "
            "and matching the question phrasing to the doc text."
        )

    out["final"] = f_json
    return out
