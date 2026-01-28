# a2a_protocol.py
# ------------------------------------------------------------
# Defines a simple Agent-to-Agent (A2A) Protocol Schema
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any
import json


@dataclass
class A2AMessage:
    role: str
    type: str
    content: Dict[str, Any]
    timestamp: str

    @classmethod
    def create(
        cls,
        role: str,
        type_: str,
        content: Dict[str, Any],
    ) -> "A2AMessage":
        return cls(
            role=role,
            type=type_,
            content=content,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "A2AMessage":
        data = json.loads(text)

        # FIX: Handle both nested ('content') and flat A2A message structures
        if "content" not in data:
            content = {
                k: v
                for k, v in data.items()
                if k not in ("role", "type", "timestamp")
            }
            data["content"] = content

        # Fill missing timestamp if executor omitted it
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Clean extra fields to match constructor
        return cls(
            role=data.get("role", ""),
            type=data.get("type", ""),
            content=data.get("content", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        )
