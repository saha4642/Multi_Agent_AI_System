# longterm_memory.py

import os
import json
import math
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_FILE = "longterm_memory.json"


def _load_vectors() -> List[Dict]:
    """Load all stored memory vectors."""
    if not os.path.exists(VECTOR_FILE):
        return []
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def _save_vectors(vectors: List[Dict]) -> None:
    """Save all memory vectors back to disk."""
    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)


def cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def embed_text(text: str):
    """Convert text into a vector embedding using OpenAI embeddings API."""
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[embed] Error: {e}")
        return []


def store_summary(session_id: str, summary: str):
    """Store a summary and its embedding into long-term memory."""
    vectors = _load_vectors()
    emb = embed_text(summary)
    if not emb:
        print("[longterm] Skipped storing summary (embedding failed).")
        return

    # ADDED: stable id + type + print includes id
    mem_id = f"mem:{len(vectors):06d}"

    vectors.append(
        {
            "id": mem_id,
            "type": "summary",
            "session": session_id,
            "summary": summary,
            "embedding": emb,
        }
    )

    _save_vectors(vectors)
    print(f"[longterm] Stored summary from session {session_id} as id={mem_id}")


# ADDED
def store_note(text: str, session_id: str = "global", tags: Optional[List[str]] = None):
    """Store a freeform note into long-term memory (used in hybrid RAG)."""
    vectors = _load_vectors()
    emb = embed_text(text)
    if not emb:
        print("[longterm] Skipped storing note (embedding failed).")
        return

    mem_id = f"mem:{len(vectors):06d}"

    vectors.append(
        {
            "id": mem_id,
            "type": "note",
            "session": session_id,
            # kept same field name for compatibility with existing code
            "summary": text,
            "embedding": emb,
            "tags": tags or [],
        }
    )

    _save_vectors(vectors)
    print(f"[longterm] Stored note id={mem_id}")


def recall_relevant(query: str, top_k: int = 2):
    """Return top-k relevant summaries for the given query based on cosine similarity."""
    vectors = _load_vectors()
    if not vectors:
        return []

    q_emb = embed_text(query)
    if not q_emb:
        return []

    scored = []
    for v in vectors:
        emb = v.get("embedding") or []
        if emb:
            scored.append((v, cosine(q_emb, emb)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [v.get("summary", "") for v, _ in scored[:top_k] if v.get("summary")]


# ADDED
def search_memory(query: str, top_k: int = 3) -> List[Dict]:
    """
    Semantic search across all memory vectors (summaries + notes).
    Returns: [{"id","text","score","type","session"}, ...]
    """
    vectors = _load_vectors()
    if not vectors:
        return []

    q_emb = embed_text(query)
    if not q_emb:
        return []

    scored: List[Dict] = []
    for v in vectors:
        emb = v.get("embedding") or []
        if not emb:
            continue
        s = cosine(q_emb, emb)
        scored.append(
            {
                "id": v.get("id", ""),
                "text": v.get("summary", ""),
                "score": float(s),
                "type": v.get("type", "summary"),
                "session": v.get("session", ""),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
