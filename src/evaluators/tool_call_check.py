"""tool_call_check: 에이전트가 실제로 호출한 도구 목록을 시나리오의
required/forbidden 리스트와 비교하는 규칙 기반 평가기.

판단 기준:
- required: 시나리오에 명시된 모든 도구가 최소 1회 호출되어야 pass
- forbidden: 시나리오에 명시된 도구가 단 1회라도 호출되면 fail

입력은 LangGraph의 invoke() 결과에서 나오는 messages 리스트를 그대로 받는다.
"""
from __future__ import annotations

from typing import Any


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


def tool_call_check(messages: list[Any], expected_tool_calls: dict) -> dict:
    """단일 시나리오에 대한 tool_call_check 평가.

    Args:
        messages: build_graph().invoke(...)["messages"]
        expected_tool_calls: scenarios.json의 해당 시나리오 expected_tool_calls
            {"required": [...], "forbidden": [...]}

    Returns:
        {"passed": bool, "detail": str, "called_tools": list[str]}
    """
    called = extract_called_tools(messages)
    called_set = set(called)

    required = expected_tool_calls.get("required", [])
    forbidden = expected_tool_calls.get("forbidden", [])

    missing_required = [t for t in required if t not in called_set]
    violated_forbidden = [t for t in forbidden if t in called_set]

    passed = not missing_required and not violated_forbidden

    if passed:
        detail = f"필수 도구 {required} 모두 호출, 금지 도구 미호출"
    else:
        parts = []
        if missing_required:
            parts.append(f"필수 도구 미호출: {missing_required}")
        if violated_forbidden:
            parts.append(f"금지 도구 호출됨: {violated_forbidden}")
        detail = " / ".join(parts)

    return {
        "passed": passed,
        "detail": detail,
        "called_tools": called,
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