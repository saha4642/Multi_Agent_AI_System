# main.py

import os
import argparse
import asyncio
import json
from dotenv import load_dotenv


from agents import Agent, Runner, SQLiteSession
from agent_config import build_agent

from history_utils import add_history, print_help
from summarizer_agent import summarize_history
from summaries_utils import save_summary
from longterm_memory import store_summary, recall_relevant

from document_embedder import generate_doc_embedding, list_documents, get_document_text
from vector_store import VectorStore
from vs_utils import bootstrap_from_document_embeddings

# =========================
# RAG imports
# =========================
from ingestion import ingest_file, ingest_dir, list_docs, get_doc, delete_doc
from qa_agent import answer_with_context  # document-only QA

# =========================
# Hybrid imports
# =========================
from hybrid_retriever import build_hybrid_context
from qa_hybrid_agent import answer_hybrid
from longterm_memory import store_note, search_memory

# =========================
# Evaluation imports
# =========================
from eval_runner import run_single_eval
from evaluator import load_results, summarize_results

# =========================
# MCP context imports
# =========================
from mcp_context import build_mcp_context
from qa_mcp_agent import answer_with_mcp_context

# =========================
# Added - External MCP tools + context
# =========================
from mcp_external_tools import TOOLS as MCP_TOOLS
from mcp_external_tools import list_external_cache, clear_external_cache
from external_context import build_mcp_context_with_external
from qa_mcp_ext_agent import answer_with_mcp_external

from interagent_chat import interagent_chat_demo  # <-- new import
from a2a_share_demo import a2a_share_demo  # <-- new import
from multi_agent_collab import run_multi_agent_collab  # <-- new import

load_dotenv()

DEFAULT_TOP_K = 5


async def run_turn(agent: Agent, user_text: str, session: SQLiteSession) -> str:
    result = await Runner.run(agent, input=user_text, session=session)
    return getattr(result, "final_output", str(result))


def banner():
    print("===============================================")
    print("Use AgentKit Sessions to Simulate Inter-Agent Chat")  # Added
    print("===============================================")

def menu():
    print("Commands:")
    print("  /embed_doc                  Create an embedding for a document (legacy)")
    print("  /list_docs                  Show legacy embedded documents (legacy)")
    print("  /show_doc <name>            View legacy doc content (legacy)")
    print("  /recall                     Recall from long-term memory (summaries)")
    print("  /summarize                  Summarize current chat")
    print("  /vs_use <local|pinecone>    Switch active vector store backend")
    print("  /vs_bootstrap               Index all docs from document_embeddings.json (legacy)")
    print("  /vs_upsert                  Upsert a single text into the vector store")
    print("  /vs_search                  Semantic search in the vector store")
    print("  /rag_ingest_file <path>     Ingest a .txt/.md/.pdf file into the vector store")
    print("  /rag_ingest_dir <path>      Ingest all supported files in a folder")
    print("  /rag_docs                   List ingested documents (manifest)")
    print("  /rag_doc <doc_id>           Show details for an ingested doc")
    print("  /rag_delete <doc_id>        Remove doc from manifest (keeps vectors)")
    print("  /ask [question]             Ask a question; uses retrieval + QA")
    print("  /mem_add                    Add a freeform memory note (long-term)")
    print("  /mem_search                 Semantic search in memory")
    print("  /hyb_ask                    Ask with HYBRID context (Memory + Docs)")
    print("  /eval_new                   Evaluate one Q/A with Hybrid Context")
    print("  /eval_report                Show aggregate eval metrics")
    print("  /mcp_ctx_preview            Build + preview MCP context for a query")
    print("  /mcp_ask                    Ask using MCP context fields")

    # ===== Added: EXTERNAL knowledge via MCP tools =====
    print("  /ext_tools                  List available MCP external tools")
    print("  /ext_fetch_wiki             Fetch wikipedia article (stores chunks)")
    print("  /ext_fetch_url              Fetch arbitrary URL (stores chunks)")
    print("  /ext_fetch_readme           Fetch GitHub README (owner/repo)")
    print("  /ext_list                   List cached external chunks")
    print("  /ext_clear                  Clear cached external chunks")
    print("  /mcp_ext_preview            Build + preview MCP context (with external)")
    print("  /mcp_ask_ext                Ask using MCP context incl. EXTERNAL")

    print("  /inter_chat                Run inter-agent chat demo")  # Added
    print("  /a2a_share                 Run A2A protocol schema demo")  # Added
    print("  /multi_query               Hands-On: Test Multi-Agent Knowledge Queries")  # Added
    print("  /help                       Show commands")
    print("  exit / quit                 Quit program")
    print("-----------------------------------------------\n")


def _read_multiline(prompt: str) -> str:
    print(prompt)
    print("(end with ENTER + Ctrl+Z on Windows / Ctrl+D on Mac/Linux)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your .env file")

    parser = argparse.ArgumentParser(description="Hands-On: Local Vector Store + Hybrid Context + Evaluation + MCP + External MCP")
    parser.add_argument("--session-id", default="doc_embed_session", help="Session ID for reasoning")
    parser.add_argument("--db-path", default="agent_memory.sqlite", help="SQLite DB path for session memory")
    parser.add_argument("--vs-backend", default="local", help="Vector store backend: local | pinecone")
    args = parser.parse_args()

    session = SQLiteSession(session_id=args.session_id, db_path=args.db_path)
    agent = build_agent()
    history = []

    backend = (args.vs_backend or "local").lower()
    try:
        vs = VectorStore(backend=backend)
    except Exception as e:
        print(f"[vs] Falling back to local backend due to error: {e}")
        vs = VectorStore(backend="local")

    banner()
    print(f"[init] Active Vector Store: {vs.name()}\n")
    menu()

    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            # Convenience: auto-add "/" for these prefixes if user forgets it
            if (
                user_text.startswith("rag_")
                or user_text.startswith("vs_")
                or user_text.startswith("mem_")
                or user_text.startswith("hyb_")
                or user_text.startswith("eval_")
                or user_text.startswith("mcp_")
                or user_text.startswith("ext_")
            ):
                user_text = "/" + user_text

            if user_text.lower() in ("exit", "quit"):
                print(f"\nBye! Resume with: python main.py --session-id {args.session_id}")
                break

            if user_text.startswith("/"):
                parts = user_text.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/help":
                    print_help()
                    menu()
                    continue

                # -------------------------
                # Legacy document embedder
                # -------------------------
                if cmd == "/embed_doc":
                    doc_name = input("Enter document name: ").strip()
                    doc_text = _read_multiline("Paste or type your document text:")
                    if not doc_text:
                        print("[embed_doc] Empty text; nothing saved.\n")
                        continue
                    generate_doc_embedding(doc_text, doc_name)
                    continue

                if cmd == "/list_docs":
                    list_documents()
                    continue

                if cmd == "/show_doc":
                    if not arg:
                        print("[cmd] Usage: /show_doc <document_name>\n")
                        continue
                    text = get_document_text(arg)
                    if text:
                        print(f"\n========== {arg} ==========\n{text}\n===========================\n")
                    else:
                        print("[show_doc] Document not found.\n")
                    continue

                # -------------------------
                # Long-term summary memory
                # -------------------------
                if cmd == "/recall":
                    topic = input("Enter topic to recall: ").strip()
                    recalls = recall_relevant(topic)
                    if not recalls:
                        print("[recall] No relevant memories found.\n")
                        continue
                    print("\n[recall] Retrieved relevant summaries:")
                    for i, s in enumerate(recalls, 1):
                        print(f"{i:02d}. {s}\n")
                    continue

                if cmd == "/summarize":
                    if not history:
                        print("\n[summarize] No messages yet.\n")
                        continue
                    summary = asyncio.run(summarize_history(history))
                    save_summary(summary, args.session_id)
                    store_summary(args.session_id, summary)
                    print("[summary] Added to long-term memory.\n")
                    continue

                # -------------------------
                # Vector store commands
                # -------------------------
                if cmd == "/vs_use":
                    choice = arg.lower() if arg else input("Backend (local|pinecone): ").strip().lower()
                    try:
                        vs = VectorStore(backend=choice)
                        print(f"[vs] Switched to backend: {vs.name()}\n")
                    except Exception as e:
                        print(f"[vs] Error switching backend: {e}\n")
                    continue

                if cmd == "/vs_bootstrap":
                    n = bootstrap_from_document_embeddings(vs)
                    print(f"[vs] Bootstrapped {n} documents into '{vs.name()}' store.\n")
                    continue

                if cmd == "/vs_upsert":
                    item_id = input("Item ID (e.g., note:001): ").strip()
                    title = input("Title (metadata): ").strip()
                    text = _read_multiline("Paste the text to index:")
                    if not item_id or not text:
                        print("[vs_upsert] Missing id or text.\n")
                        continue
                    vs.upsert(item_id, text, {"title": title})
                    continue

                if cmd == "/vs_search":
                    query = input("Search query: ").strip()
                    topk_raw = input("Top K (default 5): ").strip()
                    try:
                        topk = int(topk_raw) if topk_raw else 5
                    except ValueError:
                        topk = 5

                    results = vs.query(query, top_k=topk)
                    if not results:
                        print("[vs] No results.\n")
                        continue

                    print("\n========== Vector Search Results ==========")
                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata", {}) or {}
                        title = meta.get("title") or meta.get("name") or meta.get("path") or "--"
                        score = r.get("score", 0.0)
                        rid = r.get("id", "--")
                        print(f"{i:02d}. id={rid} | score={score:.4f} | title={title}")
                    print("==========================================\n")
                    continue

                # =========================
                # RAG commands
                # =========================
                if cmd == "/rag_ingest_file":
                    path = arg or input("File path: ").strip()
                    if not path:
                        print("[rag] Usage: /rag_ingest_file <path>\n")
                        continue
                    try:
                        doc_id = ingest_file(path, vs)
                        print(f"[rag] Ingested file -> doc_id={doc_id}\n")
                    except Exception as e:
                        print(f"[rag] Ingest failed: {e}\n")
                    continue

                if cmd == "/rag_ingest_dir":
                    path = arg or input("Directory path: ").strip()
                    if not path:
                        print("[rag] Usage: /rag_ingest_dir <path>\n")
                        continue
                    try:
                        doc_ids = ingest_dir(path, vs)
                        print(f"[rag] Ingested {len(doc_ids)} docs.\n")
                    except Exception as e:
                        print(f"[rag] Ingest dir failed: {e}\n")
                    continue

                if cmd == "/rag_docs":
                    try:
                        docs = list_docs()
                        if not docs:
                            print("[rag] No ingested docs.\n")
                            continue
                        print("\n========== Ingested Docs ==========")
                        for i, d in enumerate(docs, 1):
                            did = d.get("doc_id") or d.get("id") or "--"
                            title = d.get("title") or d.get("name") or d.get("path") or "--"
                            n_chunks = d.get("num_chunks") or d.get("chunks") or d.get("n_chunks") or "--"
                            print(f"{i:02d}. {did} | {title} | chunks={n_chunks}")
                        print("===================================\n")
                    except Exception as e:
                        print(f"[rag] List docs failed: {e}\n")
                    continue

                if cmd == "/rag_doc":
                    doc_id = arg or input("doc_id: ").strip()
                    if not doc_id:
                        print("[rag] Usage: /rag_doc <doc_id>\n")
                        continue
                    try:
                        d = get_doc(doc_id)
                        if not d:
                            print("[rag] Not found.\n")
                            continue
                        print("\n========== Doc Details ==========")
                        for k, v in d.items():
                            print(f"{k}: {v}")
                        print("=================================\n")
                    except Exception as e:
                        print(f"[rag] Get doc failed: {e}\n")
                    continue

                if cmd == "/rag_delete":
                    doc_id = arg or input("doc_id: ").strip()
                    if not doc_id:
                        print("[rag] Usage: /rag_delete <doc_id>\n")
                        continue
                    try:
                        delete_doc(doc_id)
                        print("[rag] Deleted from manifest (vectors kept).\n")
                    except Exception as e:
                        print(f"[rag] Delete failed: {e}\n")
                    continue

                # =========================
                # Docs-only /ask
                # =========================
                if cmd == "/ask":
                    question = arg or input("Question: ").strip()
                    if not question:
                        print("[ask] Please enter a question.\n")
                        continue

                    try:
                        hits = vs.query(question, top_k=DEFAULT_TOP_K) or []
                    except Exception as e:
                        print(f"[ask] Vector store query failed: {e}\n")
                        hits = []

                    if hits:
                        print("\n[ask] Retrieved hits:")
                        for i, h in enumerate(hits, 1):
                            meta = h.get("metadata", {}) or {}
                            has_text = isinstance(meta, dict) and bool(meta.get("text"))
                            print(
                                f"  {i:02d}. id={h.get('id', '--')} "
                                f"| score={float(h.get('score', 0.0)):.4f} "
                                f"| has_text={has_text}"
                            )
                        print()
                    else:
                        print("[ask] No context retrieved. Answering without documents.\n")

                    contexts = []
                    for h in hits:
                        meta = h.get("metadata", {}) or {}
                        chunk_text = meta.get("text") or ""
                        if not chunk_text:
                            continue
                        contexts.append(
                            {"id": h.get("id"), "score": h.get("score", 0.0), "text": chunk_text, "metadata": meta}
                        )

                    try:
                        answer = asyncio.run(answer_with_context(question, contexts=contexts))
                        print(f"\nAnswer:\n{answer}\n")
                    except Exception as e:
                        print(f"[ask] QA failed: {e}\n")
                    continue

                # =========================
                # Memory commands
                # =========================
                if cmd == "/mem_add":
                    note = input("Memory note to store: ").strip()
                    sess = input("Session id for this note (default 'global'): ").strip() or "global"
                    tags = input("Comma-separated tags (optional): ").strip()
                    tag_list = [t.strip() for t in tags.split(",")] if tags else []
                    store_note(note, session_id=sess, tags=tag_list)
                    continue

                if cmd == "/mem_search":
                    q = input("Search memory for: ").strip()
                    topk_in = input("Top K (default 3): ").strip()
                    try:
                        topk = int(topk_in) if topk_in else 3
                    except ValueError:
                        topk = 3

                    hits = search_memory(q, top_k=topk)
                    if not hits:
                        print("[mem] No memory hits.\n")
                        continue

                    print("\n========== Memory Search ==========")
                    for i, h in enumerate(hits, 1):
                        print(
                            f"{i:02d}. id={h['id']} | score={h['score']:.4f} | type={h['type']} | session={h['session']}"
                        )
                        txt = h.get("text", "")
                        print(f"    {txt[:160]}{'...' if len(txt) > 160 else ''}")
                    print("===================================\n")
                    continue

                # =========================
                # Hybrid ask (Memory + Docs)
                # =========================
                if cmd == "/hyb_ask":
                    query = input("Your question: ").strip()
                    topk_docs_in = input("Top K document chunks (default 5): ").strip()
                    topk_mem_in = input("Top K memory items (default 3): ").strip()
                    filter_doc = input("Filter by doc_id (optional): ").strip() or None

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    ctx = build_hybrid_context(
                        query=query,
                        store=vs,
                        top_k_docs=topk_docs,
                        top_k_mem=topk_mem,
                        filter_doc_id=filter_doc,
                    )
                    mem_items = ctx["memory"]
                    doc_items = ctx["docs"]

                    print("\n[hybrid] Memory hits:")
                    for i, m in enumerate(mem_items, 1):
                        print(
                            f"  {i:02d}. id={m.get('id','')} | score={m.get('score',0.0):.4f} "
                            f"| type={m.get('type','')} | session={m.get('session','')}"
                        )

                    print("\n[hybrid] Document hits:")
                    for i, d in enumerate(doc_items, 1):
                        meta = d.get("metadata", {}) or {}
                        doc = meta.get("doc_id", "--")
                        title = meta.get("title", "--")
                        print(
                            f"  {i:02d}. id={d.get('id','')} | score={d.get('score',0.0):.4f} "
                            f"| doc={doc} | title={title}"
                        )

                    ans = asyncio.run(answer_hybrid(query, mem_items, doc_items))
                    print(f"\nAnswer:\n{ans}\n")
                    continue

                # =========================
                # Evaluation: one-shot Q/A scoring
                # =========================
                if cmd == "/eval_new":
                    question = input("Question: ").strip()
                    gold = input("Gold (reference) answer: ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 3): ").strip()
                    filter_doc = input("Filter by doc_id (optional): ").strip() or None

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    rec = asyncio.run(
                        run_single_eval(
                            question=question,
                            gold_answer=gold,
                            store=vs,
                            top_k_docs=topk_docs,
                            top_k_mem=topk_mem,
                            filter_doc_id=filter_doc,
                            extra_meta={"session_id": args.session_id},
                        )
                    )

                    m = rec["metrics"]
                    print("\n========== Evaluation ==========")
                    print(f"Q:    {question}")
                    print(f"Gold: {gold}")
                    print(f"Answer: {rec['answer']}\n")
                    print("Scores:")
                    print(f"  semantic_sim       : {m['semantic_sim']}")
                    print(f"  token_f1           : {m['token_f1']}")
                    print(f"  has_citation       : {m['has_citation']}")
                    print(f"  citation_precision : {m['citation_precision']}")
                    print(f"  supported          : {m['supported']}")
                    print(f"  length_weight      : {m['length_weight']}")
                    print(f"  OVERALL            : {m['overall']}")
                    print(f"Citations: {', '.join(rec['citations']['cited_ids']) or '-'}")
                    print(f"Supported: {', '.join(rec['citations']['supported_ids']) or '-'}")
                    print("================================\n")
                    continue

                # =========================
                # Evaluation: aggregate report
                # =========================
                if cmd == "/eval_report":
                    rows = load_results()
                    if not rows:
                        print("[eval] No evaluation results yet. Run /eval_new first.\n")
                        continue
                    agg = summarize_results(rows)
                    print("\n========== Evaluation Report ==========")
                    for k, v in agg.items():
                        print(f"{k:<18}: {v}")
                    print(f"items_evaluated     : {len(rows)}")
                    print("======================================\n")
                    continue

                # =========================
                # MCP context preview
                # =========================
                if cmd == "/mcp_ctx_preview":
                    query = input("Query to build MCP context for: ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 3): ").strip()
                    filter_doc = input("Filter by doc_id (optional): ").strip() or None

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    user_profile = {"role": "developer", "course": "AI Personal Assistant System Design"}  # demo stub
                    ctx = build_mcp_context(
                        query=query,
                        session=session,
                        store=vs,
                        top_k_mem=topk_mem,
                        top_k_docs=topk_docs,
                        filter_doc_id=filter_doc,
                        user_profile=user_profile,
                    )

                    print("\n========== MCP Context Preview ==========")
                    print(json.dumps(ctx, ensure_ascii=False, indent=2))
                    print("=========================================\n")
                    continue

                # =========================
                # Ask using MCP context fields
                # =========================
                if cmd == "/mcp_ask":
                    query = input("Your question: ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 3): ").strip()
                    filter_doc = input("Filter by doc_id (optional): ").strip() or None

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    user_profile = {"role": "developer", "course": "AI Personal Assistant System Design"}  # demo stub
                    mcp = build_mcp_context(
                        query=query,
                        session=session,
                        store=vs,
                        top_k_mem=topk_mem,
                        top_k_docs=topk_docs,
                        filter_doc_id=filter_doc,
                        user_profile=user_profile,
                    )

                    ans = asyncio.run(answer_with_mcp_context(query, mcp))
                    print(f"\nAnswer:\n{ans}\n")
                    continue

                # ==========================================================
                # Added — EXTERNAL KNOWLEDGE via MCP tools
                # ==========================================================
                if cmd == "/ext_tools":
                    print("\nAvailable MCP tools:")
                    for k in MCP_TOOLS.keys():
                        print(f"- {k}")
                    print()
                    continue

                if cmd == "/ext_fetch_wiki":
                    topic = arg or input("Wikipedia title: ").strip()
                    try:
                        tool = MCP_TOOLS["wikipedia"]
                        docs = tool.fetch(title=topic)  # type: ignore
                        print(f"[ext] Stored {len(docs)} wiki chunks for '{topic}'.\n")
                    except Exception as e:
                        print(f"[ext] Error: {e}\n")
                    continue

                if cmd == "/ext_fetch_url":
                    url = arg or input("URL: ").strip()
                    try:
                        tool = MCP_TOOLS["http"]
                        docs = tool.fetch(url=url)  # type: ignore
                        print(f"[ext] Stored {len(docs)} chunks from URL.\n")
                    except Exception as e:
                        print(f"[ext] Error: {e}\n")
                    continue

                if cmd == "/ext_fetch_readme":
                    repo = arg or input("GitHub repo (owner/name): ").strip()
                    try:
                        tool = MCP_TOOLS["github_readme"]
                        docs = tool.fetch(repo=repo)  # type: ignore
                        print(f"[ext] Stored {len(docs)} README chunks for '{repo}'.\n")
                    except Exception as e:
                        print(f"[ext] Error: {e}\n")
                    continue

                if cmd == "/ext_list":
                    items = list_external_cache()
                    if not items:
                        print("[ext] No external items cached.\n")
                        continue
                    print("\n========== External Cache ==========")
                    for i, it in enumerate(items, 1):
                        _id = it.get("id", "--")
                        src = it.get("source", "--")
                        title = (it.get("title") or "")[:50]
                        print(f"{i:02d}. id={_id} | source={src} | title={title}")
                    print("===================================\n")
                    continue

                if cmd == "/ext_clear":
                    clear_external_cache()
                    print("[ext] External cache cleared.\n")
                    continue

                # ==========================================================
                # Added — MCP context with EXTERNAL cache included
                # ==========================================================
                if cmd == "/mcp_ext_preview":
                    query = input("Query to build MCP (with external): ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 3): ").strip()

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    user_profile = {"role": "developer"}
                    ctx = build_mcp_context_with_external(
                        query=query,
                        session=session,
                        store=vs,
                        top_k_mem=topk_mem,
                        top_k_docs=topk_docs,
                        user_profile=user_profile,
                    )

                    print("\n========== MCP Context + External ==========")
                    print(json.dumps(ctx, ensure_ascii=False, indent=2))
                    print("============================================\n")
                    continue

                if cmd == "/mcp_ask_ext":
                    query = input("Your question: ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 3): ").strip()

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 3
                    except ValueError:
                        topk_mem = 3

                    user_profile = {"role": "developer"}
                    mcp = build_mcp_context_with_external(
                        query=query,
                        session=session,
                        store=vs,
                        top_k_mem=topk_mem,
                        top_k_docs=topk_docs,
                        user_profile=user_profile,
                    )

                    ans = asyncio.run(answer_with_mcp_external(query, mcp))
                    print(f"\nAnswer:\n{ans}\n")
                    continue

                
                # ==========================================================
                # Added — Inter-agent chat demo (/inter_chat)
                # ==========================================================
                if cmd == "/inter_chat":
                    asyncio.run(interagent_chat_demo())
                    continue

                
                # ==========================================================
                # Added — A2A protocol schema demo (/a2a_share)
                # ==========================================================
                if cmd == "/a2a_share":
                    asyncio.run(a2a_share_demo())
                    continue


                # ==========================================================
                # Added — Multi-agent collab demo (/multi_query)
                # ==========================================================
                if cmd == "/multi_query":
                    query = input("Enter your question: ").strip()
                    topk_docs_in = input("Top K docs (default 5): ").strip()
                    topk_mem_in = input("Top K memory (default 2): ").strip()

                    try:
                        topk_docs = int(topk_docs_in) if topk_docs_in else 5
                    except ValueError:
                        topk_docs = 5

                    try:
                        topk_mem = int(topk_mem_in) if topk_mem_in else 2
                    except ValueError:
                        topk_mem = 2

                    asyncio.run(
                        run_multi_agent_collab(
                            vs,
                            session,
                            query=query,
                            topk_docs=topk_docs,
                            topk_mem=topk_mem,
                        )
                    )
                    continue

                print("[cmd] Unknown command. Type /help for options.\n")
                continue

            # ==== Regular chat ====
            add_history(history, "user", user_text)
            reply = asyncio.run(run_turn(agent, user_text, session))
            add_history(history, "assistant", reply)
            print(f"\nAgent: {reply}\n")

    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume session: {args.session_id}")


if __name__ == "__main__":
    main()