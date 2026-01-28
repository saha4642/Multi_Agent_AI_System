# a2a_schema.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import datetime


def utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# =============================================================================
# Base A2A message
# =============================================================================

@dataclass
class A2AMessage:
    """
    Base class for any A2A JSON message (tolerant to extra keys).
    """

    role: str
    type: str
    content: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=utc_now)

    # Accept unknown keys without breaking
    def __init__(self, **kwargs):
        self.role = kwargs.get("role", "")
        self.type = kwargs.get("type", "")
        self.content = kwargs.get("content", {})
        self.timestamp = kwargs.get("timestamp", utc_now())

        # Preserve any unknown keys
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)


# =============================================================================
# TaskSpec: Planner → Executor
# =============================================================================

@dataclass
class TaskSpec(A2AMessage):
    """
    Task specification message from Planner → Executor.
    """
    content: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Result: Executor → Planner
# =============================================================================

@dataclass
class Result(A2AMessage):
    """
    Structured result message from Executor → Planner.
    """
    status: str = "success"
    output: str = ""
    sources: Optional[List[str]] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)
