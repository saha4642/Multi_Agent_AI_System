# pinecone_vector_store.py

import os
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from longterm_memory import embed_text

# -------------------------------------------------------------------
# Defaults (can be overridden when instantiating PineconeVectorStore)
# -------------------------------------------------------------------
DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "edge-embeddings")
DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_DIM = 1536  # text-embedding-3-small uses 1536 dims


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = DEFAULT_INDEX,
        cloud: str = DEFAULT_CLOUD,
        region: str = DEFAULT_REGION,
    ):
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        self._ensure_index(index_name, cloud, region)
        self.index = self.pc.Index(self.index_name)

    # ------------------------------------------------------------------
    def name(self) -> str:
        return "pinecone"

    # ------------------------------------------------------------------
    def _ensure_index(self, name: str, cloud: str, region: str):
        names = [ix.name for ix in self.pc.list_indexes()]
        if name not in names:
            print(
                f"[pinecone] Creating index '{name}' "
                f"(serverless {cloud}/{region})..."
            )
            self.pc.create_index(
                name=name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(
                "[pinecone] Index creation requested "
                "(it may take ~60s to become ready)."
            )

    # ------------------------------------------------------------------
    def upsert(self, item_id: str, text: str, metadata: Dict[str, Any]) -> None:
        emb = embed_text(text)
        if not emb:
            print("[pinecone] Skipping upsert (embedding failed).")
            return

        # Store raw text inside metadata for retrieval
        meta_out = (metadata or {}).copy()
        meta_out["text"] = text

        self.index.upsert(
            vectors=[
                {
                    "id": item_id,
                    "values": emb,
                    "metadata": meta_out,
                }
            ]
        )
        print(f"[pinecone] Upserted id='{item_id}'.")

    # ------------------------------------------------------------------
    def batch_upsert(self, items: List[Dict[str, Any]]) -> None:
        to_send = []

        for it in items:
            emb = embed_text(it["text"])
            if not emb:
                print(
                    f"[pinecone] Skipping '{it.get('id')}' "
                    "(embedding failed)."
                )
                continue

            meta_out = (it.get("metadata", {}) or {}).copy()
            meta_out["text"] = it["text"]

            to_send.append(
                {
                    "id": it["id"],
                    "values": emb,
                    "metadata": meta_out,
                }
            )

        if to_send:
            self.index.upsert(vectors=to_send)
            print(f"[pinecone] Batch upserted {len(to_send)} items.")

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        q = embed_text(query_text)
        if not q:
            return []

        res = self.index.query(
            vector=q,
            top_k=top_k,
            include_metadata=True,
        )

        out: List[Dict[str, Any]] = []

        # Supports both object-style and dict-style responses
        matches = getattr(res, "matches", None) or res.get("matches", [])

        for m in matches:
            mid = (
                getattr(m, "id", None)
                or (m.get("id") if isinstance(m, dict) else None)
            )
            mscore = float(
                getattr(m, "score", 0.0)
                or (m.get("score", 0.0) if isinstance(m, dict) else 0.0)
            )
            mmeta = (
                getattr(m, "metadata", None)
                or (m.get("metadata", {}) if isinstance(m, dict) else {})
            )

            out.append(
                {
                    "id": mid,
                    "score": mscore,
                    "metadata": mmeta,
                    # NOTE: text is available as metadata["text"]
                }
            )

        return out
