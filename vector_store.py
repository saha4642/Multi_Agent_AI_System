# vector_store.py

from typing import List, Dict, Any, Optional
from local_vector_store import LocalVectorStore

try:
    from pinecone_vector_store import PineconeVectorStore  # optional
except Exception:
    PineconeVectorStore = None


class VectorStore:
    """
    Thin facade that lets you switch between 'local' and 'pinecone' backends.
    """

    def __init__(self, backend: str = "local", **kwargs):
        backend = backend.lower().strip()
        if backend == "pinecone":
            if PineconeVectorStore is None:
                raise RuntimeError(
                    "Pinecone backend requested but pinecone client not available. "
                    "Install pinecone-client and set env vars."
                )
            self.impl = PineconeVectorStore(**kwargs)
        else:
            self.impl = LocalVectorStore(**kwargs)

    def name(self) -> str:
        return self.impl.name()

    def upsert(
        self,
        item_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.impl.upsert(item_id, text, metadata or {})

    def batch_upsert(self, items: List[Dict[str, Any]]) -> None:
        """
        items: [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        self.impl.batch_upsert(items)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns: [{"id": str, "score": float, "metadata": {...}}, ...], highest score first
        """
        return self.impl.query(query_text, top_k=top_k)
