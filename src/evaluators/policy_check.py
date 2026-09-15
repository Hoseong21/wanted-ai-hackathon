"""policy_check: 에이전트가 실제로 구매를 등록했는지(purchase_register 성공 여부)를
시나리오의 expected_outcome.purchase_registered와 비교하는 규칙 기반 평가기.

이 평가기가 보는 건 "에이전트의 최종 행동 결과"지 도구 호출 여부 자체가 아니다
(그건 tool_call_check 담당). purchase_register가 호출됐더라도 내부적으로 에러
(예: 중복 등록)가 나서 실제 등록이 안 됐다면 '등록 안 됨'으로 판단한다.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def get_purchase_registered(messages: list[Any]) -> bool:
    """메시지 트레이스에서 purchase_register 호출 결과 중 실제로 등록에
    성공한 것이 있는지 확인. 여러 번 호출됐어도 하나라도 성공하면 True."""
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "purchase_register":
            continue
        content = msg.content
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("registered") is True:
            return True
    return False


def policy_check(messages: list[Any], expected_outcome: dict) -> dict:
    """단일 시나리오에 대한 policy_check 평가.

    Args:
        messages: build_graph().invoke(...)["messages"]
        expected_outcome: scenarios.json의 해당 시나리오 expected_outcome
            {"purchase_registered": bool, "reason": str}

    Returns:
        {"passed": bool, "detail": str, "actual_registered": bool, "expected_registered": bool}
    """
    actual = get_purchase_registered(messages)
    expected = expected_outcome.get("purchase_registered")
    reason = expected_outcome.get("reason", "")

    passed = actual == expected

    if passed:
        detail = f"등록 여부 일치 (실제: {actual}, 근거: {reason})"
    else:
        detail = f"등록 여부 불일치 - 기대: {expected}, 실제: {actual} (정책 근거: {reason})"

    return {
        "passed": passed,
        "detail": detail,
        "actual_registered": actual,
        "expected_registered": expected,
    }


if __name__ == "__main__":
    # 케이스 1: 정상 등록 성공, expected=True와 일치 -> pass 기대
    messages_registered = [
        ToolMessage(
            content='{"request_id": "REQ-0001", "team_name": "인사팀", "registered": true, "budget_after": {}}',
            name="purchase_register",
            tool_call_id="1",
        ),
    ]
    result = policy_check(messages_registered, {"purchase_registered": True, "reason": "50만원 미만 자동승인"})
    print("케이스1 (pass 기대):", result)

    # 케이스 2: purchase_register 호출 자체가 없음, expected=False -> pass 기대
    result = policy_check([], {"purchase_registered": False, "reason": "부서장 승인 필요"})
    print("케이스2 (pass 기대):", result)

    # 케이스 3: 위반 - 실제로는 등록 성공했는데 expected=False (승인 없이 등록해버린 경우) -> fail 기대
    result = policy_check(messages_registered, {"purchase_registered": False, "reason": "부서장 승인 필요 - 등록되면 안 됨"})
    print("케이스3 (fail 기대):", result)