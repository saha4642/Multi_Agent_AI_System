# app.py

from __future__ import annotations

import os
import base64
import json
import asyncio
import inspect
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# =============================================================================
# ENV / SECRETS
# =============================================================================

load_dotenv()


def _prime_openai_key_from_secrets() -> bool:
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            return True
    except Exception:
        pass
    return False


_prime_openai_key_from_secrets()


def api_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def run_maybe_async(value):
    if inspect.iscoroutine(value) or inspect.isawaitable(value):
        return _run_async(value)
    return value


# =============================================================================
# Project imports
# =============================================================================

from vector_store import VectorStore
from retriever import retrieve
from qa_agent import answer_with_context
from ingestion import ingest_uploaded_file, list_docs, get_doc, ingest_url  # ✅ ingest_url added

from ui_render import render_chunks, render_eval_table
from evaluator import evaluate_item

from longterm_memory import recall_relevant, store_summary
from hybrid_retriever import build_hybrid_context
from qa_hybrid_agent import answer_hybrid
from a2a_hybrid import run_a2a_hybrid
from multi_agent_collab import run_multi_agent_collab


# =============================================================================
# OpenAI fallback
# =============================================================================

def answer_from_gpt(question: str) -> str:
    if not api_key_present():
        return "OpenAI API key is missing, so I cannot answer from GPT. Add OPENAI_API_KEY in Streamlit Secrets or .env."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."},
                {"role": "user", "content": question},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"GPT fallback failed: {e}"


# =============================================================================
# Source link builder + viewer
# =============================================================================

def _build_source_link(meta: Dict[str, Any], chunk_id: str) -> str:
    """
    - For URL sources: return external deep link (section/text fragment)
    - For file sources: return in-app source viewer link
    """
    source_type = (meta.get("source_type") or "").lower()

    # ✅ URL sources => external link to exact portion
    if source_type == "url":
        return meta.get("source_url_exact") or meta.get("source_url_section") or meta.get("source_url") or meta.get("title") or ""

    # Existing behavior for PDFs/TXT
    doc_id = meta.get("doc_id") or ""
    page = meta.get("page_number")
    if page:
        return f"?view=source&doc_id={doc_id}&chunk_id={chunk_id}&page={int(page)}"
    return f"?view=source&doc_id={doc_id}&chunk_id={chunk_id}"


def _render_sources(doc_items: List[Dict[str, Any]], *, title: str = "Sources") -> None:
    st.markdown(f"### {title}")

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in doc_items:
        cid = it.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        uniq.append(it)

    if not uniq:
        st.caption("No sources available.")
        return

    for i, it in enumerate(uniq[:8], 1):
        meta = it.get("metadata", {}) or {}
        cid = it.get("id", "")

        source_type = (meta.get("source_type") or "").lower()
        page = meta.get("page_number")

        if source_type == "url":
            # Show section title + external link
            section = meta.get("section_title") or meta.get("title") or "Web source"
            link = _build_source_link(meta, cid)
            st.markdown(f"{i}. [{section}]({link})  \n`{cid}`")
        else:
            # File source (PDF/TXT)
            doc_title = meta.get("title") or meta.get("doc_id") or "document"
            chunk_idx = meta.get("chunk_index")
            link = _build_source_link(meta, cid)

            label_parts = [f"{doc_title}"]
            if page:
                label_parts.append(f"page {int(page)}")
            if isinstance(chunk_idx, int):
                label_parts.append(f"chunk {chunk_idx}")
            label = " — ".join(label_parts)

            st.markdown(f"{i}. [{label}]({link})  \n`{cid}`")


def _source_viewer() -> None:
    qp = st.query_params
    if qp.get("view") != "source":
        return

    doc_id = qp.get("doc_id", "")
    chunk_id = qp.get("chunk_id", "")
    page = qp.get("page")
    try:
        page_num = int(page) if page is not None else None
    except Exception:
        page_num = None

    st.markdown("---")
    st.markdown("## 📌 Source Viewer")

    if not doc_id or not chunk_id:
        st.info("No source selected.")
        return

    d = get_doc(doc_id)
    if not d:
        st.warning(f"Doc not found in manifest: {doc_id}")
        return

    st.markdown(f"**Document:** `{doc_id}`")
    st.markdown(f"**File:** `{d.get('source_path','')}`")
    if page_num:
        st.markdown(f"**PDF Page:** {page_num}")

    st.caption("Showing the exact retrieved chunk text stored in metadata during ingestion.")

    found_text = None
    hit_meta: Dict[str, Any] = {}

    try:
        exact = st.session_state.store.get_by_id(chunk_id)
        if exact:
            hit_meta = (exact.get("metadata", {}) or {})
            found_text = hit_meta.get("text") or exact.get("text")
    except Exception:
        exact = None

    if not found_text:
        try:
            hits = retrieve(query=chunk_id, store=st.session_state.store, top_k=20, filter_doc_id=doc_id)
            for h in hits:
                if h.get("id") == chunk_id:
                    hit_meta = (h.get("metadata", {}) or {})
                    found_text = hit_meta.get("text") or h.get("text")
                    break
        except Exception:
            found_text = None

    if not found_text:
        st.warning("Could not locate this chunk in the vector store.")
        return

    source_path = d.get("source_path", "")
    if (source_path or "").lower().endswith(".pdf") and os.path.exists(source_path):
        try:
            with open(source_path, "rb") as f:
                pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
            target_page = int(page_num or hit_meta.get("page_number") or 1)
            st.markdown(f"**Preview (PDF page {target_page})**")
            st.components.v1.html(
                f'<iframe src="data:application/pdf;base64,{pdf_b64}#page={target_page}" width="100%" height="700"></iframe>',
                height=720,
            )
        except Exception as e:
            st.info(f"PDF preview unavailable: {e}")

    st.text_area("Exact Chunk Text", value=found_text, height=260)


# =============================================================================
# App init
# =============================================================================

st.set_page_config(page_title="Doc/PDF/TXT/URL QA (RAG + GPT fallback)", layout="wide")
st.title("📚 Ask Your Uploaded Docs + Web Links (RAG) + GPT Fallback")

if "history" not in st.session_state:
    st.session_state.history = []

if "store" not in st.session_state:
    st.session_state.store = VectorStore()

store: VectorStore = st.session_state.store
api_ok = api_key_present()

# =============================================================================
# Sidebar Controls
# =============================================================================

st.sidebar.header("Configuration")

top_k_docs = st.sidebar.slider("Top-K Docs", min_value=1, max_value=10, value=5, step=1)
top_k_mem = st.sidebar.slider("Top-K Memory", min_value=0, max_value=10, value=3, step=1)
min_relevance = st.sidebar.slider("Min relevance (score)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)

filter_doc_id = st.sidebar.text_input(
    "Filter by doc_id (optional)",
    value="",
    placeholder="e.g. uploaded_0_somefile.pdf or url_0",
) or None

with st.sidebar.expander("Advanced"):
    show_sources = st.checkbox("Show retrieved chunks", value=False)
    persist_answers = st.checkbox("Store answer summary into long-term memory", value=False)
    show_debug = st.checkbox("Show debug/errors", value=True)

try:
    _docs = list_docs(store)
    st.sidebar.caption(f"Indexed docs: {len(_docs) if _docs else 0}")
except Exception:
    st.sidebar.caption("Indexed docs: unknown")

if not api_ok:
    st.sidebar.warning(
        "OPENAI_API_KEY not detected. RAG retrieval still works, but answering and GPT fallback require a key.\n\n"
        "Add it via Streamlit Secrets or a local .env."
    )

# =============================================================================
# Tabs
# =============================================================================

tab_rag, tab_hybrid, tab_a2a, tab_multi, tab_ingest, tab_memory, tab_eval = st.tabs(
    ["RAG QA", "Hybrid", "A2A Hybrid", "Multi-Agent", "Ingest", "Memory", "Eval"]
)

_source_viewer()

# =============================================================================
# RAG QA
# =============================================================================

with tab_rag:
    st.subheader("Ask Questions (Docs/Links → if not found → GPT fallback)")

    q = st.text_input("Question", placeholder="Ask something grounded in your uploaded docs or ingested links", key="q_rag")

    if st.button("Run", type="primary"):
        try:
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving from indexed content..."):
                    doc_items = retrieve(
                        store=store,
                        query=q,
                        top_k=top_k_docs,
                        filter_doc_id=filter_doc_id,
                    )

                top_score = float(doc_items[0].get("score", 0.0)) if doc_items else 0.0
                has_doc_answer = bool(doc_items) and top_score >= float(min_relevance)

                if show_sources and doc_items:
                    render_chunks(doc_items, title="Retrieved Chunks (debug view)")

                if has_doc_answer:
                    with st.spinner("Answering from your indexed content..."):
                        ans = run_maybe_async(answer_with_context(q, doc_items))

                    st.markdown("### Answer (from files/links)")
                    st.write(ans)

                    _render_sources(doc_items, title="Sources (click to open exact portion)")

                    st.session_state.history.append(("RAG(files+links)", q, ans, {"docs": len(doc_items), "top_score": top_score}))

                    if persist_answers and api_ok:
                        store_summary("global", f"Q: {q}\nA: {ans}")
                        st.success("Saved a brief summary into long-term memory.")
                else:
                    st.warning("I don’t have the answer from the indexed files/links.")
                    with st.spinner("Answering with GPT..."):
                        gpt_ans = answer_from_gpt(q)

                    st.markdown("### Answer (GPT)")
                    st.write(gpt_ans)

                    st.session_state.history.append(("GPT(fallback)", q, gpt_ans, {"docs": len(doc_items), "top_score": top_score}))

        except Exception as e:
            st.error("RAG flow failed.")
            if show_debug:
                st.exception(e)

# =============================================================================
# Hybrid / A2A / Multi-Agent / Memory / Eval
# (left unchanged from your current version)
# =============================================================================

with tab_hybrid:
    st.subheader("Hybrid: Documents + Long-term Memory")
    qh = st.text_input("Question", placeholder="Ask something that benefits from both docs and memory", key="q_hybrid")

    if st.button("Run Hybrid", type="primary"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not qh.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Building hybrid context..."):
                    doc_items, mem_items = build_hybrid_context(
                        store=store,
                        query=qh,
                        top_k_docs=top_k_docs,
                        top_k_mem=top_k_mem,
                        filter_doc_id=filter_doc_id,
                    )

                if not doc_items and not mem_items:
                    st.info('No context found. Ingest documents/links in **"Ingest"** and/or store summaries in **"Memory"**.')
                else:
                    if show_sources:
                        if doc_items:
                            render_chunks(doc_items, title="Doc/Link Chunks")
                        if mem_items:
                            render_chunks(mem_items, title="Memory Snippets")

                    with st.spinner("Answering..."):
                        ans = answer_hybrid(qh, doc_items, mem_items)

                    st.markdown("### Answer")
                    st.write(ans)

                    if doc_items:
                        _render_sources(doc_items, title="Sources (docs/links)")

                    st.session_state.history.append(("Hybrid", qh, ans, {"docs": len(doc_items), "mem": len(mem_items)}))

        except Exception as e:
            st.error("Hybrid flow failed.")
            if show_debug:
                st.exception(e)

with tab_a2a:
    st.subheader("A2A Hybrid: Planner → Executor (with RAG + Memory)")
    context_hint = st.text_area("Optional context for the Planner", placeholder="Add a short context blurb for the A2A exchange.", key="a2a_hint")
    qa = st.text_input("Question", placeholder="Ask a question", key="q_a2a")

    if st.button("Run A2A Hybrid", type="primary"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not qa.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running A2A Hybrid..."):
                    out = run_maybe_async(run_a2a_hybrid(store=store, session=None, query=qa, topk_docs=top_k_docs, topk_mem=top_k_mem))
                st.markdown("### Output")
                st.write(out)
        except Exception as e:
            st.error("A2A run failed.")
            if show_debug:
                st.exception(e)

with tab_multi:
    st.subheader("Multi-Agent Collaboration")
    qm = st.text_input("Question", placeholder="Ask a question", key="q_multi")

    if st.button("Run Multi-Agent", type="primary"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not qm.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running multi-agent collaboration..."):
                    out = run_maybe_async(run_multi_agent_collab(store=store, session=None, query=qm, topk_docs=top_k_docs, topk_mem=top_k_mem))
                st.markdown("### Output")
                st.write(out)
        except Exception as e:
            st.error("Multi-agent run failed.")
            if show_debug:
                st.exception(e)

# =============================================================================
# Ingest (UPDATED)
# =============================================================================

with tab_ingest:
    st.subheader("Upload & Ingest Documents (txt / md / pdf / docx)")

    uploads = st.file_uploader(
        "Upload one or more files",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )

    default_doc_id_prefix = st.text_input("doc_id prefix (optional)", value="uploaded")

    if st.button("Ingest Uploaded Files", type="primary") and uploads:
        try:
            ingested_total = 0
            for i, f in enumerate(uploads):
                doc_id = f"{default_doc_id_prefix}_{i}_{f.name}"
                with st.spinner(f"Ingesting {f.name} → {doc_id}"):
                    count = ingest_uploaded_file(file=f, doc_id=doc_id, store=store)
                ingested_total += int(count or 0)

            st.success(f"Ingested {ingested_total} chunks.")
        except Exception as e:
            st.error("Ingestion failed.")
            if show_debug:
                st.exception(e)

    st.markdown("---")
    st.subheader("Ingest Web Links (Wikipedia / Scholarpedia) ✅")

    urls_text = st.text_area(
        "Paste URLs (one per line)",
        height=120,
        placeholder="https://en.wikipedia.org/wiki/Reactive_ion_etching\nhttps://www.scholarpedia.org/article/Nuclear_force",
    )
    url_prefix = st.text_input("URL doc_id prefix", value="url")

    if st.button("Ingest Links", type="primary"):
        try:
            urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
            if not urls:
                st.warning("Paste at least one URL.")
            else:
                total_chunks = 0
                for i, u in enumerate(urls):
                    did = f"{url_prefix}_{i}"
                    with st.spinner(f"Ingesting link → {did}"):
                        ingest_url(u, store=store, doc_id=did)
                    # manifest count includes chunks; we display doc count here
                st.success(f"Ingested {len(urls)} link(s). You can now ask questions in RAG QA.")
        except Exception as e:
            st.error("Link ingestion failed (site might block bots or require JS).")
            if show_debug:
                st.exception(e)

    st.markdown("### Existing Docs / Links")
    try:
        docs = list_docs(store)
        if docs:
            st.write(docs)
            st.caption("Tip: filter by doc_id in the sidebar (e.g., url_0) if you want to restrict answers to one page.")
        else:
            st.info("No docs indexed yet.")
    except Exception as e:
        st.warning("Could not list docs.")
        if show_debug:
            st.exception(e)

# =============================================================================
# Memory / Eval remain unchanged
# =============================================================================

with tab_memory:
    st.subheader("Memory Tools")
    prompt = st.text_input("Ask memory to recall:", placeholder="e.g., What summaries do we have about retrieval?")
    if st.button("Recall"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not prompt.strip():
                st.warning("Please enter a recall prompt.")
            else:
                with st.spinner("Recalling..."):
                    mem_items = recall_relevant(prompt, top_k=top_k_mem)
                if not mem_items:
                    st.info("No memory items found yet.")
                else:
                    render_chunks(mem_items, title="Recalled Memory")
        except Exception as e:
            st.error("Memory recall failed.")
            if show_debug:
                st.exception(e)

    st.markdown("---")
    to_store = st.text_area("Store a short summary to long-term memory")
    if st.button("Store Summary"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not to_store.strip():
                st.warning("Please enter some text to store.")
            else:
                store_summary("global", to_store)
                st.success("Stored.")
        except Exception as e:
            st.error("Failed to store summary.")
            if show_debug:
                st.exception(e)

with tab_eval:
    st.subheader("Quick Eval")
    q_eval = st.text_input("Question (eval)")
    gold = st.text_area("Gold Answer")

    if st.button("Run Eval"):
        try:
            if not api_ok:
                st.error("OPENAI_API_KEY is missing. Add it to Secrets or .env.")
            elif not q_eval.strip():
                st.warning("Please enter an evaluation question.")
            else:
                with st.spinner("Retrieving..."):
                    doc_items = retrieve(store=store, query=q_eval, top_k=top_k_docs, filter_doc_id=filter_doc_id)

                if not doc_items:
                    st.info("No retrieved chunks for eval. Ingest docs/links first.")
                else:
                    with st.spinner("Answering..."):
                        ans = run_maybe_async(answer_with_context(q_eval, doc_items))

                    st.markdown("### Model Answer")
                    st.write(ans)

                    rec = evaluate_item(
                        question=q_eval,
                        gold=gold,
                        answer=ans,
                        contexts=doc_items,
                        meta={"top_k_docs": top_k_docs, "filter_doc_id": filter_doc_id},
                    )

                    st.markdown("### Eval Result")
                    render_eval_table(rec)

        except Exception as e:
            st.error("Eval failed.")
            if show_debug:
                st.exception(e)

with st.sidebar.expander("History", expanded=False):
    if st.session_state.history:
        for mode, q_, a_, meta in reversed(st.session_state.history[-50:]):
            st.markdown(f"**[{mode}]** `{q_}`")
            st.caption(json.dumps(meta))
            st.write(a_)
            st.markdown("---")
    else:
        st.caption("No interactions yet.")
