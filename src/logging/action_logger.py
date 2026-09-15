"""실행 로그.

verify_and_execute 노드가 게이트를 거친 액션(현재는 purchase_register)의 판정 과정을
JSONL로 한 줄씩 append한다. 실험 결과 집계의 원천이자, 데모에서 "감사 추적(audit trail)"으로 쓴다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "actions.jsonl"


def log_action(
    *,
    thread_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    gate_decision: str,
    reason: str,
    human_decision: str | None = None,  # "approved" | "rejected" | None(승인 불필요한 경우)
    executed: bool,
    scenario_id: str | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "scenario_id": scenario_id,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "gate_decision": gate_decision,
        "reason": reason,
        "human_decision": human_decision,
        "executed": executed,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")