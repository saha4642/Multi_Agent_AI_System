# summaries_utils.py

import json
import os
from typing import List

# Folder to store all summaries
SUMMARY_DIR = "summaries"
SUMMARY_FILE_SUFFIX = "_conversation_summaries.jsonl"


def _get_summary_file(session_id: str) -> str:
    """Return the full file path for the session summary file."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    return os.path.join(
        SUMMARY_DIR,
        f"{session_id}{SUMMARY_FILE_SUFFIX}"
    )


def save_summary(summary_text: str, session_id: str) -> None:
    """Append a summary entry to the file for this session."""
    filepath = _get_summary_file(session_id)
    record = {
        "session": session_id,
        "summary": summary_text
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[saved] Summary stored in {filepath}")


def load_summaries(session_id: str) -> List[str]:
    """Load all summaries for the given session."""
    filepath = _get_summary_file(session_id)

    if not os.path.exists(filepath):
        return []

    summaries = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "summary" in data:
                    summaries.append(data["summary"])
            except json.JSONDecodeError:
                continue

    return summaries


def show_summaries(session_id: str) -> None:
    """Print all summaries for a given session in readable form."""
    summaries = load_summaries(session_id)

    if not summaries:
        print(
            f"\n[summaries] No summaries found for session "
            f"'{session_id}'.\n"
        )
        return

    print(
        f"\n========== Stored Summaries for {session_id} =========="
    )
    for i, summary in enumerate(summaries, 1):
        print(f"{i:02d}. {summary}\n")
    print("======================================================\n")
