# memory_backends.py

from agents import SQLiteSession

# Redis session support may differ depending on SDK version

def make_sqlite_session(
    db_path: str = "agent_memory.sqlite",
    session_id: str | None = None
):
    return SQLiteSession(
        session_id=session_id,
        db_path=db_path
    )
