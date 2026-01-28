# mcp_external_tools.py
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# ---- Optional dependency -----------------------------------------------------
try:
    import requests  # type: ignore
except Exception:
    requests = None


# ---- Data model --------------------------------------------------------------

@dataclass
class ExtDoc:
    id: str
    title: str
    text: str
    source: str
    url: str = ""
    meta: Optional[Dict[str, Any]] = None


class ExternalTool(ABC):
    """
    Abstract base for external fetchers that return a list of ExtDoc chunks.
    """
    name: str = "external"

    @abstractmethod
    def fetch(self, *args, **kwargs) -> List[ExtDoc]:
        ...


# ---- Chunking helpers --------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    """
    Collapse excessive whitespace while preserving paragraph breaks.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 120,
) -> List[str]:
    """
    Simple character-based chunker that prefers to break on
    sentence/paragraph boundaries near the max length.
    Overlap helps with context continuity.
    """
    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        return [text]

    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        # Try to end at a boundary near "end"
        boundary = max(
            text.rfind("\n\n", start, end),
            text.rfind(". ", start, end),
            text.rfind("! ", start, end),
            text.rfind("? ", start, end),
        )

        # If no good boundary or too close to start, hard cut
        if boundary == -1 or boundary <= start + max(50, overlap):
            boundary = end
        else:
            boundary += 1  # include trailing punctuation/space

        piece = text[start:boundary].strip()
        if piece:
            pieces.append(piece)

        start = max(boundary - overlap, start + 1)

    return pieces


# ---- Cache helpers -----------------------------------------------------------

_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".external_cache.json")


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"items": []}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _cache_store(docs: List[ExtDoc]) -> None:
    cache = _load_cache()
    now = time.time()
    for d in docs:
        cache["items"].append(
            {
                "id": d.id,
                "title": d.title,
                "text": d.text,
                "source": d.source,
                "url": d.url,
                "meta": d.meta or {},
                "added_at": now,
            }
        )
    _save_cache(cache)


def list_external_cache() -> List[Dict[str, Any]]:
    """Return all cached items as plain dicts."""
    cache = _load_cache()
    return cache.get("items", [])


def clear_external_cache() -> None:
    """Clear the on-disk external cache."""
    _save_cache({"items": []})


# ---- HTTP helper -------------------------------------------------------------

def _http_get(url: str, *, params=None, timeout: int = 20):
    if requests is None:
        raise RuntimeError("requests not available. Please pip install requests.")
    headers = {
        "User-Agent": "Rohan-AI-Demo/1.0 (https://example.com/contact)",
        "Accept": "application/json,text/plain,*/*",
    }
    return requests.get(url, params=params, headers=headers, timeout=timeout)


# ---- Generic HTTP fetch tool -------------------------------------------------

class HttpFetchTool(ExternalTool):
    name = "http"

    def fetch(self, url: str) -> List[ExtDoc]:
        if requests is None:
            raise RuntimeError("requests not available. Please pip install requests.")

        r = _http_get(url, timeout=30)
        r.raise_for_status()

        # Accept text or JSON
        content_type = r.headers.get("Content-Type", "")
        if "application/json" in content_type:
            content = json.dumps(r.json(), ensure_ascii=False)
        else:
            content = r.text

        content = _normalize_whitespace(content)
        chunks = _chunk(content)

        out: List[ExtDoc] = []
        for i, c in enumerate(chunks):
            out.append(
                ExtDoc(
                    id=f"ext:http:{i:04d}",
                    title=url,
                    text=c,
                    source="http",
                    url=url,
                    meta={"chunk_index": i},
                )
            )

        _cache_store(out)
        return out


# ---- Wikipedia tool ----------------------------------------------------------

class WikipediaTool(ExternalTool):
    name = "wikipedia"

    def fetch(self, title: str, lang: str = "en") -> List[ExtDoc]:
        """
        Fetch a concise summary for a Wikipedia title.
        1) Try REST summary API (clean, short).
        2) Fallback to MediaWiki Action API (plaintext extract).
        """
        if requests is None:
            raise RuntimeError("requests not available. Please pip install requests.")

        from urllib.parse import quote

        slug = quote(title.replace(" ", "_"))

        # 1) Try REST summary API
        rest_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            r = _http_get(rest_url)
            if r.status_code == 200:
                data = r.json()
                txt = "\n".join(
                    [
                        data.get("title", ""),
                        data.get("description", ""),
                        data.get("extract", ""),
                    ]
                ).strip()

                if txt:
                    page_url = (
                        data.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", f"https://{lang}.wikipedia.org/wiki/{slug}")
                    )
                    chunks = _chunk(txt)
                    out: List[ExtDoc] = []
                    for i, c in enumerate(chunks):
                        out.append(
                            ExtDoc(
                                id=f"ext:wikipedia:{data.get('title','untitled')}:{i:04d}",
                                title=data.get("title", "wikipedia"),
                                text=c,
                                source="wikipedia",
                                url=page_url,
                                meta={"chunk_index": i, "api": "rest_v1"},
                            )
                        )
                    _cache_store(out)
                    return out
        except Exception:
            # Intentionally ignore and try fallback
            pass

        # 2) Fallback to MediaWiki Action API (plaintext extract)
        action_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
            "origin": "*",
        }

        r2 = _http_get(action_url, params=params)
        r2.raise_for_status()
        j = r2.json()
        pages = j.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            raise RuntimeError(f"Wikipedia page not found for '{title}'.")

        page = pages[0]
        txt = (page.get("extract") or "").strip()
        if not txt:
            raise RuntimeError("Empty extract from action API.")

        page_title = page.get("title", title)
        page_url = f"https://{lang}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}"

        chunks = _chunk(txt)
        out: List[ExtDoc] = []
        for i, c in enumerate(chunks):
            out.append(
                ExtDoc(
                    id=f"ext:wikipedia:{page_title}:{i:04d}",
                    title=page_title,
                    text=c,
                    source="wikipedia",
                    url=page_url,
                    meta={"chunk_index": i, "api": "action"},
                )
            )

        _cache_store(out)
        return out


# ---- GitHub README tool ------------------------------------------------------

class GitHubReadmeTool(ExternalTool):
    name = "github_readme"

    def fetch(self, repo: str) -> List[ExtDoc]:
        """
        Fetch README.md for a GitHub repo via raw.githubusercontent.com.
        repo format: "owner/name"
        Tries HEAD, main, then master.
        """
        if requests is None:
            raise RuntimeError("requests not available. Please pip install requests.")

        candidates = [
            f"https://raw.githubusercontent.com/{repo}/HEAD/README.md",
            f"https://raw.githubusercontent.com/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{repo}/master/README.md",
        ]

        content = None
        src = None
        for u in candidates:
            try:
                r = requests.get(u, timeout=20)
                if r.status_code == 200 and r.text.strip():
                    content = r.text
                    src = u
                    break
            except Exception:
                continue

        if not content:
            raise RuntimeError(f"README not found for repo '{repo}'.")

        chunks = _chunk(content)
        out: List[ExtDoc] = []
        for i, c in enumerate(chunks):
            out.append(
                ExtDoc(
                    id=f"ext:github:{repo}:{i:04d}",
                    title=f"{repo} README.md",
                    text=c,
                    source="github",
                    url=src or "",
                    meta={"chunk_index": i},
                )
            )

        _cache_store(out)
        return out


# ---- Registry ----------------------------------------------------------------

TOOLS: Dict[str, ExternalTool] = {
    "http": HttpFetchTool(),
    "wikipedia": WikipediaTool(),
    "github_readme": GitHubReadmeTool(),
}


__all__ = [
    "ExtDoc",
    "ExternalTool",
    "HttpFetchTool",
    "WikipediaTool",
    "GitHubReadmeTool",
    "TOOLS",
    "list_external_cache",
    "clear_external_cache",
]
