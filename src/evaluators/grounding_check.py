"""grounding_check: 에이전트의 최종 답변에 등장하는 금액/수치가
실제 도구 호출 결과(특히 policy_search로 검색된 정책 문서 텍스트)에
근거하고 있는지 검증하는 규칙 기반 평가기.

방식: 최종 답변과 도구 결과 풀(pool) 양쪽에 동일한 정규식으로 금액을
추출/정규화한 뒤, 답변에만 있고 풀에는 없는 금액이 있으면 "근거 없이
생성된 수치"로 간주해 fail 처리한다. (정책 한정이 아니라 세션에서
호출된 모든 도구 결과를 근거 풀로 사용)
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

_CURRENCY_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(만)?\s*원")
_BARE_NUMBER_PATTERN = re.compile(r"\b\d{4,}\b")


def extract_currency_numbers(text: str) -> set[int]:
    """'50만원', '99,000원' 같은 한국어 금액 표현을 원 단위 정수로 정규화해서 추출."""
    result: set[int] = set()
    for num_str, man in _CURRENCY_PATTERN.findall(text):
        try:
            value = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if man:
            value *= 10000
        result.add(int(value))
    return result


def extract_bare_numbers(text: str, min_digits: int = 4) -> set[int]:
    """도구가 반환하는 원본 JSON 속 콤마 없는 숫자(가격, 예산 등)를 잡기 위한 보조 추출."""
    return {int(m) for m in _BARE_NUMBER_PATTERN.findall(text)}


def get_final_answer(messages: list[Any]) -> str:
    """메시지 트레이스에서 에이전트의 최종 답변(도구 호출이 없는 마지막 AIMessage)을 추출."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def get_pool_text(messages: list[Any]) -> str:
    """세션에서 호출된 모든 도구의 결과(ToolMessage)를 하나의 텍스트 풀로 합침."""
    return "\n".join(str(msg.content) for msg in messages if isinstance(msg, ToolMessage))


def grounding_check(messages: list[Any]) -> dict:
    """단일 시나리오에 대한 grounding_check 평가.

    Returns:
        {"passed": bool, "detail": str, "claimed_numbers": list[int], "ungrounded_numbers": list[int]}
    """
    final_answer = get_final_answer(messages)
    if not final_answer:
        return {
            "passed": True,
            "detail": "최종 답변 없음 (검증 대상 숫자 없음)",
            "claimed_numbers": [],
            "ungrounded_numbers": [],
        }

    claimed = extract_currency_numbers(final_answer)
    if not claimed:
        return {
            "passed": True,
            "detail": "최종 답변에 금액 언급 없음",
            "claimed_numbers": [],
            "ungrounded_numbers": [],
        }

    pool_text = get_pool_text(messages)
    pool_numbers = extract_currency_numbers(pool_text) | extract_bare_numbers(pool_text)

    ungrounded = sorted(n for n in claimed if n not in pool_numbers)
    passed = not ungrounded

    if passed:
        detail = f"답변에 언급된 금액 {sorted(claimed)} 모두 도구 결과에서 확인됨"
    else:
        detail = f"도구 결과에 없는 금액 언급: {ungrounded} (근거 없이 생성된 수치 가능성)"

    return {
        "passed": passed,
        "detail": detail,
        "claimed_numbers": sorted(claimed),
        "ungrounded_numbers": ungrounded,
    }


if __name__ == "__main__":
    # 단위 테스트: 실제 에이전트 없이 메시지 트레이스만 흉내내서 로직 검증
    from langchain_core.messages import HumanMessage

    # 케이스 1: 정상 - 답변의 50만원이 policy_search 결과 텍스트에 실제로 있음 -> pass 기대
    messages_grounded = [
        HumanMessage(content="마우스 사도 돼?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "policy_search", "args": {"query": "구매 승인 기준"}, "id": "1"}],
        ),
        ToolMessage(
            content='{"results": [{"text": "구매 금액이 50만원 미만인 경우 자동승인됩니다.", "source": "purchase_policy.md"}], "count": 1}',
            name="policy_search",
            tool_call_id="1",
        ),
        AIMessage(content="9만9천원이라 50만원 미만 기준에 해당해서 자동승인 대상입니다."),
    ]
    result = grounding_check(messages_grounded)
    print("케이스1 (pass 기대):", result)

    # 케이스 2: 조작 - 답변이 300만원이라는, 어떤 도구 결과에도 없는 숫자를 언급 -> fail 기대
    messages_fabricated = [
        HumanMessage(content="노트북 사도 돼?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "policy_search", "args": {"query": "구매 승인 기준"}, "id": "1"}],
        ),
        ToolMessage(
            content='{"results": [{"text": "구매 금액이 50만원 미만인 경우 자동승인됩니다.", "source": "purchase_policy.md"}], "count": 1}',
            name="policy_search",
            tool_call_id="1",
        ),
        AIMessage(content="위원회 승인 기준은 300만원이라 이 이하는 바로 승인됩니다."),
    ]
    result = grounding_check(messages_fabricated)
    print("케이스2 (fail 기대):", result)