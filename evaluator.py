# evaluator.py

import json
import os
import re
import statistics
from typing import List, Dict, Any, Tuple

from longterm_memory import embed_text  # uses your existing embedding helper

EVAL_FILE = "eval_results.jsonl"

_CITATION_PATTERNS = [
    r"\(source:\s*([^)]+)\)",     # (source: doc#0001)
    r"\(doc:\s*([^)]+)\)",        # (doc: doc#0001)
    r"\(mem:\s*([^)]+)\)",        # (mem: mem:000123)
]


def _write_json(rec: Dict[str, Any], path: str = EVAL_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def token_f1(pred: str, ref: str) -> float:
    """
    Simple bag-of-words F1 (precision/recall harmonic mean).
    """
    p = _tokenize(pred)
    r = _tokenize(ref)
    if not p or not r:
        return 0.0

    p_counts, r_counts = {}, {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1

    overlap = 0
    for t, c in p_counts.items():
        if t in r_counts:
            overlap += min(c, r_counts[t])

    prec = overlap / max(len(p), 1)
    rec = overlap / max(len(r), 1)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


def semantic_similarity(pred: str, ref: str) -> float:
    """
    Cosine similarity via your embedding helper.
    """
    pe = embed_text(pred) or []
    re_ = embed_text(ref) or []
    if not pe or not re_:
        return 0.0

    dot = sum(x * y for x, y in zip(pe, re_))
    np = (sum(x * x for x in pe) ** 0.5) + 1e-9
    nr = (sum(x * x for x in re_) ** 0.5) + 1e-9
    return float(dot / (np * nr))


def extract_citations(answer: str) -> List[str]:
    hits = []
    for pat in _CITATION_PATTERNS:
        for m in re.findall(pat, answer, flags=re.IGNORECASE):
            hits.append(m.strip())

    # normalize split for "A, B"
    out = []
    for h in hits:
        parts = [p.strip() for p in re.split(r"[,;\s]+", h) if p.strip()]
        out.extend(parts or [h])

    # remove trailing punctuation
    out = [re.sub(r"[).,;:]+$", "", x) for x in out]

    # dedupe
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    return uniq


def grounding_metrics(answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check that citations in answer point to retrieved context IDs.
    """
    cited = extract_citations(answer)
    ctx_ids = {c.get("id") for c in contexts}
    good = [cid for cid in cited if cid in ctx_ids]

    metrics = {
        "has_citation": 1.0 if cited else 0.0,
        "citation_count": len(cited),
        "citation_precision": (len(good) / len(cited)) if cited else 0.0,
        "supported": 1.0 if good else 0.0,
        "cited_ids": cited,
        "supported_ids": good,
    }
    return metrics


def length_penalty(answer: str, ref: str, max_ratio: float = 2.5) -> float:
    """
    Soft penalty if answer is excessively long vs reference.
    """
    la, lr = max(len(answer), 1), max(len(ref), 1)
    ratio = la / lr
    return 1.0 if ratio <= max_ratio else max(0.4, max_ratio / ratio)


def aggregate_score(
    sim: float,
    f1: float,
    grounding_sup: float,
    has_cite: float,
    length_w: float,
) -> float:
    """
    Weighted geometric-ish mean to reward balanced performance.
    Tunings:
      semantic 0.45
      f1       0.25
      grounding 0.20
      citation 0.05
      length    0.05
    """
    w = {
        "sim": 0.45,
        "f1": 0.25,
        "ground": 0.20,
        "cite": 0.05,
        "len": 0.05,
    }

    eps = 1e-6
    parts = [
        max(sim, eps) ** w["sim"],
        max(f1, eps) ** w["f1"],
        max(grounding_sup, eps) ** w["ground"],
        max(has_cite, eps) ** w["cite"],
        max(length_w, eps) ** w["len"],
    ]

    score = 1.0
    for p in parts:
        score *= p
    return float(score)


def evaluate_item(
    question: str,
    gold: str,
    answer: str,
    contexts: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    sim = semantic_similarity(answer, gold)
    f1 = token_f1(answer, gold)
    gm = grounding_metrics(answer, contexts)
    lp = length_penalty(answer, gold)

    overall = aggregate_score(
        sim,
        f1,
        gm["supported"],
        gm["has_citation"],
        lp,
    )

    rec = {
        "question": question,
        "gold": gold,
        "answer": answer,
        "metrics": {
            "semantic_sim": round(sim, 4),
            "token_f1": round(f1, 4),
            "has_citation": gm["has_citation"],
            "citation_precision": round(gm["citation_precision"], 4),
            "supported": gm["supported"],
            "length_weight": round(lp, 4),
            "overall": round(overall, 4),
        },
        "citations": {
            "cited_ids": gm["cited_ids"],
            "supported_ids": gm["supported_ids"],
        },
        "meta": meta,
    }

    _write_json(rec)
    return rec


def load_results(path: str = EVAL_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = [
        "semantic_sim",
        "token_f1",
        "citation_precision",
        "supported",
        "overall",
    ]

    acc = {k: [] for k in keys}
    for r in rows:
        m = r.get("metrics", {})
        for k in keys:
            if k in m:
                acc[k].append(float(m[k]))

    return {
        f"avg_{k}": round(statistics.mean(v), 4)
        for k, v in acc.items()
        if v
    }
