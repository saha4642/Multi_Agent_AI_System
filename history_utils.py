# history_utils.py

import time
import json
from typing import List, Dict, Any, Optional


def now_ts() -> float:
    return time.time()


def format_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def add_history(
    history: List[Dict[str, Any]],
    role: str,
    text: str,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    history.append({
        "ts": now_ts(),
        "role": role,
        "text": text,
        "meta": meta or {}
    })


def print_history(
    history: List[Dict[str, Any]],
    last_n: Optional[int] = None
) -> None:
    if not history:
        print("\n[history] No messages yet.\n")
        return

    items = history[-last_n:] if last_n else history

    print("\n========== Conversation History ==========")
    for i, msg in enumerate(items, 1):
        print(
            f"{i:02d}. "
            f"[{format_ts(msg['ts'])}] "
            f"{msg['role']}: {msg['text']}"
        )
    print("==========================================\n")


def find_in_history(
    history: List[Dict[str, Any]],
    keyword: str
) -> None:
    keyword = keyword.lower()

    matches = [
        (i, m)
        for i, m in enumerate(history, 1)
        if keyword in m["text"].lower()
    ]

    if not matches:
        print(f"\n[find] No matches for '{keyword}'.\n")
        return

    print(f"\n[find] Matches for '{keyword}':")
    for i, msg in matches:
        print(
            f"- #{i:02d} "
            f"[{format_ts(msg['ts'])}] "
            f"{msg['role']}: {msg['text']}"
        )
    print()


def export_history_jsonl(
    history: List[Dict[str, Any]],
    filepath: str
) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in history:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    print(f"\n[export] Saved history to {filepath}\n")


def print_help() -> None:
    print("""
Commands:
/history            Show full conversation history
/history N          Show last N messages
/find <keyword>     Search messages for a keyword
/export <file.jsonl> Export history to a JSONL file
/clear              Clear memory (new session)
/help               Show this help
exit / quit         Exit the program
""")
