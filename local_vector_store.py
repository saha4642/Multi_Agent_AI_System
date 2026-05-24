# local_vector_store.py

import os
import json
import math
from typing import List, Dict, Any

from longterm_memory import embed_text  # reusing your existing embedding helper

LOCAL_VS_FILE = "local_vector_store.json"
EMBED_DIM = 1536  # text-embedding-3-small


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(x * x for x in b)) + 1e-9
    return dot / (na * nb)


def _load() -> List[Dict[str, Any]]:
    if not os.path.exists(LOCAL_VS_FILE):
        return []

    with open(LOCAL_VS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(rows: List[Dict[str, Any]]) -> None:
    with open(LOCAL_VS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


class LocalVectorStore:
    """
    Minimal local vector store for demos:
    - Stores {"id","text","embedding","metadata"} rows in a JSON file.
    - Cosine similarity search in-process.
    """

    def __init__(self, path: str = LOCAL_VS_FILE):
        self.path = path

    def name(self) -> str:
        return "local"

    def upsert(
        self,
        item_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        emb = embed_text(text)
        if not emb:
            print("[local-vs] Skipping upsert (embedding failed).")
            return

        rows = _load()
        # replace if same id exists
        rows = [r for r in rows if r.get("id") != item_id]
        rows.append({
            "id": item_id,
            "text": text,
            "embedding": emb,
            "metadata": metadata or {},
        })
        _save(rows)
        print(f"[local-vs] Upserted id='{item_id}' (dim={len(emb)}).")

    def batch_upsert(self, items: List[Dict[str, Any]]) -> None:
        rows = _load()
        existing = {r["id"]: r for r in rows if "id" in r}

        for it in items:
            emb = embed_text(it["text"])
            if not emb:
                print(f"[local-vs] Skipping '{it.get('id')}' (embedding failed).")
                continue
            existing[it["id"]] = {
                "id": it["id"],
                "text": it["text"],
                "embedding": emb,
                "metadata": it.get("metadata", {}),
            }

        _save(list(existing.values()))
        print(f"[local-vs] Batch upserted {len(items)} items.")


    def get_by_id(self, item_id: str) -> Dict[str, Any] | None:
        rows = _load()
        for r in rows:
            if r.get("id") == item_id:
                return {"id": r.get("id"), "text": r.get("text"), "metadata": r.get("metadata", {})}
        return None
    def query(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        q = embed_text(query_text)
        if not q:
            return []

        rows = _load()
        scored = []
        for r in rows:
            s = _cosine(q, r["embedding"])
            scored.append({
                "id": r["id"],
                "score": float(s),
                "metadata": r.get("metadata", {}),
                "text": r.get("text"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
