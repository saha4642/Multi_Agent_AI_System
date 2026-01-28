# chunking.py
from typing import List, Dict

def split_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[Dict]:
    """
    Sliding-window chunking with overlap.
    Returns list of {"id", "text", "metadata"} dicts.
    """
    text = text or ""
    n = len(text)
    chunks = []
    start = 0
    idx = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunk_id = f"{doc_id}#{idx:04d}"

        chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                "doc_id": doc_id,
                "chunk_index": idx,
                "start": start,
                "end": end,
            }
        })

        idx += 1
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks
