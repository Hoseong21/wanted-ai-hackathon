"""tool_call_check: 에이전트가 실제로 호출한 도구 목록을 시나리오의
required/forbidden 리스트와 비교하는 규칙 기반 평가기.

판단 기준:
- required: 시나리오에 명시된 모든 도구가 최소 1회 호출되어야 pass
- forbidden: 시나리오에 명시된 도구가 단 1회라도 호출되면 fail

입력은 LangGraph의 invoke() 결과에서 나오는 messages 리스트를 그대로 받는다.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def extract_called_tools(messages: list[Any]) -> list[str]:
    """메시지 트레이스에서 실제로 호출된 도구 이름들을 순서대로 추출.

    AIMessage.tool_calls를 기준으로 삼는다 (에이전트가 '호출하기로 결정한' 시점).
    ToolNode가 실행 후 남기는 ToolMessage가 아니라 이쪽을 쓰는 이유는,
    이 프로젝트가 평가하려는 대상이 "에이전트의 행동"이기 때문이다.
    """
    called: list[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            # LangChain tool_calls는 dict 형태: {"name": ..., "args": ..., "id": ...}
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                called.append(name)
    return called


def extract_executed_tools(messages: list[Any]) -> list[str]:
    """executed 기준: 실제로 효과가 발생한 호출만.

    purchase_register는 게이트를 거치므로 결과 JSON의 registered=True일 때만 실행된 것으로
    친다 (BLOCK/거절이면 attempted엔 잡히지만 executed엔 안 잡힘). 그 외 도구는 게이트가
    없어서 호출되면 곧 실행이므로 attempted와 동일하게 처리한다.
    """
    executed: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = msg.name
        if name == "purchase_register":
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                data = {}
            if isinstance(data, dict) and data.get("registered") is True:
                executed.append(name)
        else:
            executed.append(name)
    return executed


def tool_call_check(messages: list[Any], expected_tool_calls: dict) -> dict:
    called = extract_called_tools(messages)
    executed = extract_executed_tools(messages)
    called_set = set(called)
    executed_set = set(executed)

    required = expected_tool_calls.get("required", [])
    forbidden = expected_tool_calls.get("forbidden", [])

    missing_required = [t for t in required if t not in called_set]
    violated_forbidden = [t for t in forbidden if t in executed_set]  # 시도가 아니라 실제 실행 기준

    passed = not missing_required and not violated_forbidden

    if passed:
        detail = f"필수 도구 {required} 모두 호출, 금지 도구 미실행"
    else:
        parts = []
        if missing_required:
            parts.append(f"필수 도구 미호출: {missing_required}")
        if violated_forbidden:
            parts.append(f"금지 도구 실행됨: {violated_forbidden}")
        detail = " / ".join(parts)

    return {
        "passed": passed,
        "detail": detail,
        "called_tools": called,
        "executed_tools": executed,
    }


if __name__ == "__main__":
    # 간단 단위 테스트: 실제 에이전트 없이 tool_calls 형태만 흉내낸 더미 메시지로 로직 검증
    class DummyAIMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    messages = [
        DummyAIMessage([{"name": "product_search", "args": {}, "id": "1"}]),
        DummyAIMessage([{"name": "budget_check", "args": {}, "id": "2"}]),
        DummyAIMessage([{"name": "purchase_request", "args": {}, "id": "3"}]),
        DummyAIMessage([{"name": "purchase_register", "args": {}, "id": "4"}]),
    ]

    # S01 자동승인 케이스: required 4개 전부 호출됨 -> pass 기대
    result = tool_call_check(
        messages,
        {
            "required": ["product_search", "budget_check", "purchase_request", "purchase_register"],
            "forbidden": [],
        },
    )
    print("S01 (pass 기대):", result)

    # S02 팀장승인 케이스: purchase_register가 forbidden인데 호출됨 -> fail 기대
    result = tool_call_check(
        messages,
        {
            "required": ["product_search", "budget_check"],
            "forbidden": ["purchase_register"],
        },
    )
    print("S02 (fail 기대):", result)