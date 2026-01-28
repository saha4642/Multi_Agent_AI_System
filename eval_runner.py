# eval_runner.py

from typing import Optional, Dict, Any, List

from vector_store import VectorStore
from hybrid_retriever import build_hybrid_context
from qa_hybrid_agent import answer_hybrid
from evaluator import evaluate_item


async def run_single_eval(
    question: str,
    gold_answer: str,
    store: VectorStore,
    top_k_docs: int = 5,
    top_k_mem: int = 3,
    filter_doc_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds hybrid context, gets an answer, scores it, and logs JSONL.
    Returns the full evaluation record (including metrics).
    """

    # -------------------------
    # Build hybrid context
    # -------------------------
    ctx = build_hybrid_context(
        query=question,
        store=store,
        top_k_docs=top_k_docs,
        top_k_mem=top_k_mem,
        filter_doc_id=filter_doc_id,
    )

    mem_items: List[Dict[str, Any]] = ctx["memory"]
    doc_items: List[Dict[str, Any]] = ctx["docs"]

    # -------------------------
    # Produce answer with hybrid agent
    # -------------------------
    ans = await answer_hybrid(question, mem_items, doc_items)

    # -------------------------
    # Score and persist
    # -------------------------
    meta = {
        "top_k_docs": top_k_docs,
        "top_k_mem": top_k_mem,
        "filter_doc_id": filter_doc_id,
    }
    if extra_meta:
        meta.update(extra_meta)

    rec = evaluate_item(
        question=question,
        gold=gold_answer,
        answer=ans,
        contexts=doc_items,  # ground citations against document chunks
        meta=meta,
    )

    return rec
