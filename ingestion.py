# ingestion.py

from __future__ import annotations

import os
import time
import json
import tempfile
import shutil
import re
import html as _html
from typing import Dict, List, Tuple, Optional

from chunking import split_text
from vector_store import VectorStore

# Optional (only needed if you ingest URLs). We import lazily so file ingestion still works.
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None
    BeautifulSoup = None

MANIFEST_FILE = "docs_manifest.json"
UPLOAD_DIR = "uploaded_docs"      # persisted uploads so sources remain viewable
HTML_DIR = "html_docs"            # local 'wikipedia-like' HTML renditions

_WS_RE = re.compile(r"\s+")


def _clean_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "")).strip()


def _safe_filename(name: str) -> str:
    name = (name or "file").strip().replace("\\", "_").replace("/", "_")
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "file"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _load_manifest() -> Dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(data: Dict) -> None:
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _persist_uploaded_temp_file(tmp_path: str, *, doc_id: str, original_name: str) -> str:
    """Persist Streamlit temp upload to a stable on-disk path."""
    root = _ensure_dir(UPLOAD_DIR)
    ext = (os.path.splitext(original_name)[1] or os.path.splitext(tmp_path)[1] or ".txt").lower()
    out_name = f"{_safe_filename(doc_id)}{ext}"
    out_path = os.path.join(root, out_name)
    shutil.copyfile(tmp_path, out_path)
    return os.path.abspath(out_path)


def _read_text_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return txt, os.path.basename(path)


def _read_pdf_pages(path: str) -> Tuple[List[str], str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError("PDF support requires `pypdf` (pip install pypdf).") from e

    reader = PdfReader(path)
    pages: List[str] = []
    for p in reader.pages:
        pages.append((p.extract_text() or ""))
    return pages, os.path.basename(path)


def _read_docx_paragraphs(path: str) -> Tuple[List[str], str]:
    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise RuntimeError("DOCX support requires `python-docx` (pip install python-docx).") from e

    try:
        doc = Document(path)
    except KeyError as e:
        # python-docx raises this when the OPC package is not a valid Word document
        # (common cases: renamed .doc file, malformed/corrupt .docx).
        raise RuntimeError(
            f"Invalid DOCX structure for '{os.path.basename(path)}'. "
            "The file may be corrupted or not a true .docx document."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unable to read DOCX '{os.path.basename(path)}': {e}"
        ) from e
    paras = [((p.text or "").strip()) for p in doc.paragraphs]
    paras = [p for p in paras if p]
    return paras, os.path.basename(path)


def _write_pdf_as_wiki_html(doc_id: str, title: str, source_path: str, items: List[Dict]) -> str:
    """Create a single local HTML file with chunk anchors, like a wiki page."""
    html_root = _ensure_dir(HTML_DIR)
    out_path = os.path.join(html_root, f"{_safe_filename(doc_id)}.html")

    # Group chunks by page_number
    by_page: Dict[int, List[Dict]] = {}
    for it in items:
        meta = it.get("metadata", {}) or {}
        page = meta.get("page_number")
        page_i = int(page) if isinstance(page, int) or (isinstance(page, str) and str(page).isdigit()) else 0
        by_page.setdefault(page_i, []).append(it)

    pages = sorted(by_page.keys())

    def esc(s: str) -> str:
        return _html.escape(s or "")

    toc = []
    for p in pages:
        label = "Intro" if p == 0 else f"Page {p}"
        toc.append(f'<li><a href="#page_{p}">{esc(label)}</a></li>')

    body = []
    for p in pages:
        label = "Intro" if p == 0 else f"Page {p}"
        body.append(f'<h2 id="page_{p}">{esc(label)}</h2>')
        for it in by_page[p]:
            cid = it.get("id", "")
            meta = it.get("metadata", {}) or {}
            chunk_text = meta.get("text") or it.get("text") or ""
            body.append(
                "\n".join(
                    [
                        f'<div class="chunk" id="{esc(cid)}">',
                        '  <div class="chunkmeta">',
                        f'    <span class="chunkid">{esc(cid)}</span>',
                        '  </div>',
                        f'  <div class="chunktext">{esc(chunk_text)}</div>',
                        '</div>',
                    ]
                )
            )

    css = """
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.55; padding: 18px; max-width: 980px; margin: auto; }
      .topbar { color: #444; font-size: 14px; margin-bottom: 14px; }
      .toc { background: #f6f6f6; padding: 12px 14px; border: 1px solid #ddd; border-radius: 8px; }
      .chunk { padding: 12px; margin: 12px 0; border: 1px solid #e2e2e2; border-radius: 10px; background: #fff; }
      .chunkmeta { font-size: 12px; color: #666; margin-bottom: 8px; }
      .chunktext { white-space: pre-wrap; }
      .highlight { outline: 3px solid #f2c94c; background: rgba(242, 201, 76, 0.18); }
      a { text-decoration: none; }
      a:hover { text-decoration: underline; }
    </style>
    """

    js = """
    <script>
      (function() {
        // If Streamlit injects a target id, use it. Otherwise use window.hash.
        const target = window.__TARGET_CHUNK_ID__ || (window.location.hash ? window.location.hash.slice(1) : "");
        if (!target) return;
        const el = document.getElementById(target);
        if (!el) return;
        el.classList.add('highlight');
        el.scrollIntoView({behavior: 'instant', block: 'center'});
      })();
    </script>
    """

    html_doc = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "  <meta charset=\"utf-8\"/>",
            f"  <title>{esc(title)}</title>",
            css,
            "</head>",
            "<body>",
            f"  <h1>{esc(title)}</h1>",
            f"  <div class=\"topbar\">doc_id: <b>{esc(doc_id)}</b> — source: <code>{esc(os.path.basename(source_path))}</code></div>",
            "  <div class=\"toc\"><b>Contents</b><ul>",
            "\n".join(toc),
            "  </ul></div>",
            "\n".join(body),
            js,
            "</body>",
            "</html>",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    return os.path.abspath(out_path)


# -----------------------------
# URL ingestion (optional)
# -----------------------------

def _extract_sections_from_html(url: str, html_text: str) -> List[Dict[str, str]]:
    if BeautifulSoup is None:
        raise RuntimeError("URL ingestion requires beautifulsoup4 and lxml (pip install beautifulsoup4 lxml requests).")

    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main = soup.find(id="mw-content-text") or soup.find("article") or soup.body or soup

    sections: List[Dict[str, str]] = []
    cur_heading = "Intro"
    cur_anchor = ""
    cur_parts: List[str] = []

    def flush():
        txt = _clean_ws(" ".join(cur_parts))
        if txt:
            sections.append({"heading": cur_heading, "anchor": cur_anchor, "text": txt})

    def slug(h: str) -> str:
        return "_".join(_clean_ws(h).split())

    for el in main.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        if el.name in ("h1", "h2", "h3"):
            flush()
            cur_heading = _clean_ws(el.get_text(" ", strip=True)) or "Section"
            cur_anchor = el.get("id") or slug(cur_heading)
            cur_parts = []
        else:
            t = _clean_ws(el.get_text(" ", strip=True))
            if t:
                cur_parts.append(t)

    flush()
    return sections


def _make_text_fragment_url(base_url: str, chunk_text: str, max_words: int = 12) -> Optional[str]:
    # Best-effort exact portion highlighting in Chrome/Edge/modern browsers.
    t = _clean_ws(chunk_text)
    if not t:
        return None
    words = t.split()
    if len(words) < 4:
        return None
    frag = " ".join(words[:max_words])
    from urllib.parse import quote

    return f"{base_url}#:~:text={quote(frag, safe='')}"


def ingest_url(
    url: str,
    store: VectorStore,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 200,
    timeout_s: int = 30,
) -> str:
    """Fetch URL -> sectionize -> chunk -> upsert."""
    if requests is None:
        raise RuntimeError("URL ingestion requires requests + beautifulsoup4 + lxml.")

    url = (url or "").strip()
    if not url:
        raise ValueError("URL is empty")

    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0 (DocQA Bot)"})
    resp.raise_for_status()

    base_url = url.split("#")[0]
    sections = _extract_sections_from_html(url, resp.text)

    items: List[Dict] = []
    for si, sec in enumerate(sections):
        heading = sec.get("heading") or "Section"
        anchor = sec.get("anchor") or ""
        sec_text = sec.get("text") or ""
        if not sec_text.strip():
            continue

        section_url = f"{base_url}#{anchor}" if anchor else base_url

        chunks = split_text(sec_text, doc_id=f"{doc_id}::sec{si:03d}", chunk_size=chunk_size, overlap=overlap)
        for ch in chunks:
            exact_url = _make_text_fragment_url(section_url, ch["text"]) or section_url
            meta = (ch.get("metadata") or {}) | {
                "title": heading,
                "doc_id": doc_id,
                "source_type": "url",
                "source_url_section": section_url,
                "source_url_exact": exact_url,
                "section_title": heading,
                "text": ch["text"],
            }
            items.append({"id": ch["id"], "text": ch["text"], "metadata": meta})

    if not items:
        raise RuntimeError("No extractable text from URL.")

    store.batch_upsert(items)

    manifest = _load_manifest()
    manifest[doc_id] = {
        "doc_id": doc_id,
        "title": url,
        "source_type": "url",
        "source_url": base_url,
        "num_chunks": len(items),
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_manifest(manifest)

    print(f"[ingest_url] Ingested '{url}' as doc_id='{doc_id}' with {len(items)} chunks.")
    return doc_id


# -----------------------------
# File ingestion (existing APIs)
# -----------------------------

def ingest_file(
    path: str,
    store: VectorStore,
    doc_id: Optional[str] = None,
    chunk_size: int = 800,
    overlap: int = 200,
) -> str:
    """Read file -> chunk -> upsert into vector store.

    Changes (needed):
      - PDFs are chunked per-page and store page_number metadata.
      - PDF is also rendered into a local HTML file (wiki-like) with chunk anchors.
      - Uploads are persisted via ingest_uploaded_file (so HTML links remain valid).

    Returns the doc_id (same behavior).
    """

    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    ext = os.path.splitext(path)[1].lower()
    doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]

    items: List[Dict] = []

    if ext in {".txt", ".md"}:
        text, title = _read_text_file(path)
        if not text.strip():
            raise RuntimeError("File had no extractable text.")

        chunks = split_text(text, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap)
        for ch in chunks:
            meta = (ch.get("metadata") or {}) | {
                "title": title,
                "source_path": path,
                "source_type": ext,
                "doc_id": doc_id,
                "page_number": None,
                "text": ch["text"],
            }
            items.append({"id": ch["id"], "text": ch["text"], "metadata": meta})

        store.batch_upsert(items)

        manifest = _load_manifest()
        manifest[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "source_path": path,
            "source_type": ext,
            "num_chunks": len(items),
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_manifest(manifest)

        print(f"[ingest] Ingested '{path}' as doc_id='{doc_id}' with {len(items)} chunks.")
        return doc_id

    if ext == ".docx":
        paragraphs, title = _read_docx_paragraphs(path)
        text = "\n\n".join(paragraphs).strip()
        if not text:
            raise RuntimeError("DOCX had no extractable text.")

        chunks = split_text(text, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap)
        for ch in chunks:
            meta = (ch.get("metadata") or {}) | {
                "title": title,
                "source_path": path,
                "source_type": ext,
                "doc_id": doc_id,
                "page_number": None,
                "text": ch["text"],
            }
            items.append({"id": ch["id"], "text": ch["text"], "metadata": meta})

        store.batch_upsert(items)

        manifest = _load_manifest()
        manifest[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "source_path": path,
            "source_type": ext,
            "num_chunks": len(items),
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_manifest(manifest)

        print(f"[ingest] Ingested '{path}' as doc_id='{doc_id}' with {len(items)} chunks.")
        return doc_id

    if ext == ".pdf":
        pages, title = _read_pdf_pages(path)
        if not any((p or "").strip() for p in pages):
            raise RuntimeError("PDF had no extractable text.")

        total_chunks = 0
        for page_idx, page_text in enumerate(pages):
            if not (page_text or "").strip():
                continue
            page_num = page_idx + 1

            page_doc_id = f"{doc_id}::p{page_num:04d}"
            chunks = split_text(page_text, doc_id=page_doc_id, chunk_size=chunk_size, overlap=overlap)
            for ch in chunks:
                meta = (ch.get("metadata") or {}) | {
                    "title": title,
                    "source_path": path,
                    "source_type": ext,
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "text": ch["text"],
                }
                items.append({"id": ch["id"], "text": ch["text"], "metadata": meta})
                total_chunks += 1

        store.batch_upsert(items)

        # Generate local HTML (wiki-like) for this PDF so we can link to exact chunks
        html_path = _write_pdf_as_wiki_html(doc_id=doc_id, title=title, source_path=path, items=items)

        manifest = _load_manifest()
        manifest[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "source_path": path,
            "source_type": ext,
            "html_path": html_path,
            "num_chunks": total_chunks,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_manifest(manifest)

        print(f"[ingest] Ingested '{path}' as doc_id='{doc_id}' with {total_chunks} chunks.")
        return doc_id

    raise RuntimeError(f"Unsupported file type: {ext}")


def ingest_uploaded_file(
    file,
    doc_id: str,
    store: VectorStore,
    chunk_size: int = 800,
    overlap: int = 200,
) -> int:
    """Streamlit-friendly wrapper.

    Minimal change: persist the uploaded file so that the generated HTML and
    source links continue to work after Streamlit cleans up temp files.
    """

    suffix = os.path.splitext(getattr(file, "name", "") or "")[1].lower() or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if hasattr(file, "getbuffer"):
            tmp.write(file.getbuffer())
        else:
            tmp.write(file.read())
        tmp_path = tmp.name

    try:
        persisted_path = _persist_uploaded_temp_file(tmp_path, doc_id=doc_id, original_name=getattr(file, "name", doc_id))

        did = ingest_file(
            path=persisted_path,
            store=store,
            doc_id=doc_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        mf = _load_manifest()
        return int(mf.get(did, {}).get("num_chunks", 0))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def ingest_dir(
    folder: str,
    store: VectorStore,
    exts: tuple = (".txt", ".md", ".pdf", ".docx"),
) -> List[str]:
    """Ingest all supported files in directory tree."""

    folder = os.path.abspath(os.path.expanduser(folder))
    ingested: List[str] = []

    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                try:
                    path = os.path.join(root, name)
                    did = ingest_file(path, store)
                    ingested.append(did)
                except Exception as e:
                    print(f"[ingest] Skipped {name}: {e}")

    print(f"[ingest] Completed directory ingest. Total docs: {len(ingested)}")
    return ingested


def list_docs(store: VectorStore | None = None) -> List[Dict]:
    mf = _load_manifest()
    return list(mf.values())


def get_doc(doc_id: str) -> Dict | None:
    return _load_manifest().get(doc_id)


def delete_doc(doc_id: str, store: VectorStore | None = None) -> None:
    """Minimal delete: removes from manifest only (keeps vectors)."""
    mf = _load_manifest()
    if doc_id in mf:
        del mf[doc_id]
        _save_manifest(mf)
        print(f"[delete] Removed '{doc_id}' from manifest (vectors not deleted).")
    else:
        print(f"[delete] No doc '{doc_id}' in manifest.")
