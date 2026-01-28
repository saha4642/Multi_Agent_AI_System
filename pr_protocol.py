# pr_protocol.py

from __future__ import annotations
from typing import Any, Dict, Optional
import json


def _strip_code_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if present.
    """
    s = text.strip()
    if s.startswith("```"):
        # remove first fence line
        lines = s.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
        # remove last fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse_json_message(raw: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse a JSON object from a model/tool output string.
    Returns dict if successful, else None.
    """
    if not raw:
        return None

    s = _strip_code_fences(raw)

    # First: direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Second: try to extract the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def to_json(obj: Any) -> str:
    """
    Serialize obj to a compact JSON string.
    """
    return json.dumps(obj, ensure_ascii=False)
