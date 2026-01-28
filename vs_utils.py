# vs_utils.py
import os
import json
from typing import Any, Dict, List

# Must match document_embedder.py
DOC_EMBED_FILE = "document_embeddings.json"


def _load_docs(path: str = DOC_EMBED_FILE) -> List[Dict[str, Any]]:
    """Load docs stored by document_embedder.py."""
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def bootstrap_from_document_embeddings(vs, path: str = DOC_EMBED_FILE) -> int:
    """
    Read all stored docs from document_embeddings.json and upsert them into the active vector store.
    Expects `vs` to expose: vs.batch_upsert(items)
      where items is: [{"id": str, "text": str, "metadata": {...}}, ...]
    """
    docs = _load_docs(path)
    if not docs:
        print(f"[vs_bootstrap] No documents found in '{path}'.")
        return 0

    items: List[Dict[str, Any]] = []

    for d in docs:
        name = (d.get("name") or "untitled").strip()
        text = d.get("text") or ""
        if not text.strip():
            continue

        item_id = f"doc:{name}"

        metadata = {
            "name": name,
            "title": name,
            "source": "document_embeddings",
        }

        items.append(
            {
                "id": item_id,
                "text": text,
                "metadata": metadata,
            }
        )

    if not items:
        print(f"[vs_bootstrap] Found docs in '{path}', but none had usable text.")
        return 0

    vs.batch_upsert(items)
    print(f"[vs_bootstrap] Upserted {len(items)} documents into '{vs.name()}' store.")
    return len(items)
