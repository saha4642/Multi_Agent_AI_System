# document_embedder.py

import os
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_EMBED_FILE = "document_embeddings.json"


def _load_docs():
    """Load all stored document embeddings."""
    if not os.path.exists(DOC_EMBED_FILE):
        return []

    with open(DOC_EMBED_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_docs(docs):
    """Save updated document list."""
    with open(DOC_EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def generate_doc_embedding(text: str, doc_name: str):
    """Generate and store embedding for a document."""
    try:
        print(f"[embedding] Generating embedding for '{doc_name}'...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding

        docs = _load_docs()
        docs.append({
            "name": doc_name,
            "text": text,
            "embedding": embedding
        })

        _save_docs(docs)
        print(
            f"[embedding] Saved embedding for '{doc_name}' "
            f"({len(embedding)} dimensions)."
        )

    except Exception as e:
        print(f"[error] Failed to generate embedding: {e}")


def list_documents():
    """List all stored document embeddings."""
    docs = _load_docs()

    if not docs:
        print("[docs] No embedded documents found.\n")
        return

    print("\n========== Embedded Documents ==========")
    for i, d in enumerate(docs, 1):
        print(f"{i:02d}. {d['name']}")
    print("========================================\n")


def get_document_text(name: str):
    """Fetch the text of a stored document by name."""
    docs = _load_docs()

    for d in docs:
        if d["name"].lower() == name.lower():
            return d["text"]

    print(f"[docs] No document found with name '{name}'.\n")
    return None
