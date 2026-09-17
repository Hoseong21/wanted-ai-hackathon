"""grounding_check v3: 에이전트의 최종 답변에 등장하는 금액/수치가
'세션에서 실제로 관측된 evidence'(도구 호출 결과)에 근거하고 있는지 검증하는
규칙 기반 평가기.

세 단계로 나눠서 생각한다.
  Level 1 — 직접 근거(direct evidence): 도구 결과 JSON 필드에 그 숫자가 그대로 있음.
  Level 2 — 명시적 파생 근거(typed derived evidence): 의미가 정해진 두 라벨 사이의
            결정론적 산술 관계로 계산 가능함. 현재 두 가지를 지원한다.
              - budget_shortfall = 구매금액 - remaining_budget (구매금액 > remaining_budget일 때)
              - welfare_budget_limit = team_budget × welfare_ratio(%) / 100
                (team_budget은 budget_check 결과에서, welfare_ratio는 policy_search 결과
                원문 중 '복지/간식비' 문맥에서만 각각 실제로 관측됐을 때만 계산)
            "pool 안 아무 두 숫자의 합/차/곱을 다 허용" 같은 범용 산술 폴백은 의도적으로
            채택하지 않았다 — 매번 특정 라벨 쌍 + 명시된 관계식만 하나씩 화이트리스트로
            추가한다.
  Level 3 — 근거 없음(ungrounded): 위 둘 다 아님 -> fail.

라벨 매칭 범위 (v2 → v3에서 좁힌 부분, 매우 중요):
  "라벨: 숫자원"처럼 콜론이 숫자 바로 앞(공백/별표만 허용)에 오는, 구조가 명확한
  claim만 semantic label 검증한다. 자유 서술 문장이나 콤마로 나열된 형태
  ("배정 800만원, 사용 210만원, 잔여 590만원")는 라벨을 텍스트만으로 안전하게
  특정할 수 없으므로 라벨링을 아예 시도하지 않고 Level 1/3(존재 여부 확인)으로만
  검증한다. v2는 "숫자 앞뒤 고정폭 윈도우"와 "같은 줄 안 최근접 키워드" 두 가지
  거리 기반 휴리스틱을 순서대로 시도했었는데, 둘 다 실제 실행에서 반례가
  나왔다(각각 다른 형태로 엉뚱한 숫자에 라벨이 붙는 오탐 발생). 그래서 v3는
  "판단을 유보하고 존재 여부만 확인하는 것이 안전한 것보다 낫다"는 원칙으로
  콜론 앵커링만 남기고 프록시미티 휴리스틱을 전부 제거했다. 콤마 나열형처럼
  콜론 없이 여러 값이 나열되는 포맷은 known limitation으로 문서화한다(아래 참고).

한계 (README에 명시):
1) 콜론으로 명시적으로 연결된 claim만 라벨 검증한다. "배정 800만원, 사용 210만원,
   잔여 590만원"처럼 콤마로 나열된 형태나 콜론 없는 자유 서술문 속 숫자는 라벨
   검증을 시도하지 않고 존재 여부만 확인한다(Level 1/3). 향후 이 포맷이 자주
   관측되면 그때 별도 규칙을 추가한다 — 지금 미리 일반화하지 않는다.
2) entity-level provenance를 추적하지 않는다. 인사팀 잔여예산과 AI개발팀 잔여예산이
   세션에 둘 다 등장했다면 evidence는 팀 구분 없이 하나의 remaining_budget 집합으로
   합쳐진다.
3) budget_shortfall 파생 시 unit_price를 후보로 쓰는 건 quantity 증거가 없거나
   quantity==1일 때만 허용한다. quantity>1인데 purchase_request/purchase_register가
   한 번도 호출되지 않아 quantity 증거 자체가 없는 극단적인 경우, unit_price 기반
   shortfall이 실제 총액 기준과 다를 수 있다는 이론적 위험이 남아있다(다만
   "가격 > 잔여예산"이라는 발동 조건 자체가 대부분의 위험한 조합을 걸러낸다).
4) welfare_budget_limit은 '복지/간식비' 키워드가 있는 줄에서만 퍼센트를 채택한다.
   현재 정책 문서에는 이 문맥의 퍼센트가 10% 하나뿐이라 실질적 위험은 낮지만,
   향후 같은 문맥에 퍼센트가 여러 개 섞여 등장하면 evidence 집합이 여러 값을 담게
   되어 검증이 느슨해질 수 있다 — 그 시점에 더 좁혀야 한다.

역할 분리 원칙 (README에도 명시할 것):
    Policy Engine    → 진짜 정책이 무엇인지 판정 (authoritative truth, YAML 직접 참조 OK)
    Tool Call Check  → 필요한 정보를 실제로 조회했는지 평가
    Grounding Check  → 조회한 정보와 최종 답변이 일치하는지 평가 (본 파일)
    Outcome Check    → 최종 행동 결과가 expected_outcome과 일치하는지 평가
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

_CURRENCY_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(만)?\s*원")
_BARE_NUMBER_PATTERN = re.compile(r"\b\d{4,}\b")

# 콜론이 숫자 바로 앞(공백/별표만 허용)에 오는 구조만 매칭한다.
# 그룹1 = 콜론 앞 라벨 텍스트(최대 15자, 줄바꿈/콜론 제외), 그룹2 = 숫자, 그룹3 = "만" 여부.
_LABEL_VALUE_PATTERN = re.compile(
    r"([^\n:：]{1,15})[:：]\s*\**\s*(\d[\d,]*(?:\.\d+)?)\s*(만)?\s*원"
)

# JSON 필드명 -> 라벨. 도구 결과(ToolMessage)를 재귀적으로 훑으면서 이 표에 있는 키를
# 만나면 그 값을 해당 라벨의 evidence 집합에 넣는다. 전부 tool이 실제로 반환한 값이지
# evaluator가 계산한 값이 아니다 (judgment-free tool 원칙과 일치). "quantity"는 텍스트
# 라벨 매칭 대상이 아니라 budget_shortfall 파생 시 안전장치로만 내부적으로 쓰인다.
_FIELD_LABEL_MAP = {
    "price": "unit_price",
    "unit_price": "unit_price",
    "total_price": "total_price",
    "remaining": "remaining_budget",
    "allocated": "team_budget",
    "recent_category_total": "recent_category_total",
    "combined_total": "combined_total",
    "quantity": "quantity",
}

# 답변 텍스트에서 콜론 앞 라벨 텍스트가 이 키워드를 포함하면 해당 라벨로 채택.
#   auto_approval_limit(50만원)     = 자동승인 / 팀장승인 사이의 경계
#   manager_approval_limit(200만원) = 팀장승인 / 부서장승인 사이의 경계
#   budget_shortfall                = 구매금액 - remaining_budget (typed derivation)
#   welfare_budget_limit            = team_budget × 복지/간식비 비율 (typed derivation)
_LABEL_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("remaining_budget", ("잔여", "잔액")),
    ("team_budget", ("배정 예산", "할당 예산", "총 예산")),
    ("total_price", ("총액", "합계 금액", "총 금액", "총 구매액")),
    ("unit_price", ("단가", "가격")),
    ("recent_category_total", ("누적",)),
    ("combined_total", ("포함해", "합산")),
    ("budget_shortfall", ("부족", "부족액", "부족한")),
    ("welfare_budget_limit", ("복지", "간식비")),
    ("auto_approval_limit", ("자동승인", "자동 승인", "팀장 승인", "팀장 사전 승인")),
    ("manager_approval_limit", ("부서장", "임원", "품의")),
]

# welfare_budget_limit 파생용: pool 텍스트(도구 결과 원문) 안에서 "복지/간식비"와 같은 줄에
# 등장하는 퍼센트만 evidence로 인정한다. 임의의 퍼센트(예: 다른 규정의 숫자)가 섞여 들어오는
# 것을 막기 위한 최소한의 키워드 앵커링 — 답변 텍스트가 아니라 신뢰된 도구 결과(RAG 검색 결과)
# 원문에 대해서만 적용하므로, 최종 답변에 대한 라벨 매칭(콜론 앵커링)보다는 낮은 리스크다.
_PERCENT_PATTERN = re.compile(r"(\d{1,3})\s*%")
_WELFARE_KEYWORDS = ("복지", "간식비")


def _extract_welfare_ratio(pool_text: str) -> set[int]:
    """도구 결과 원문에서 '복지/간식비' 관련 문장에 등장하는 비율(%)만 추출."""
    ratios: set[int] = set()
    for line in pool_text.split("\n"):
        if any(kw in line for kw in _WELFARE_KEYWORDS):
            for m in _PERCENT_PATTERN.finditer(line):
                ratios.add(int(m.group(1)))
    return ratios
# 콜론 뒤 값 근처(라벨 텍스트 자체 또는 숫자 뒤 20자)에 범위 표현이 있으면
# "50만원 이상 200만원 미만"류일 가능성이 있으므로 approval_limit 라벨링은 보류한다.
_RANGE_HINTS = ("이상", "미만", "초과", "이하", "~")
_APPROVAL_LABELS = {"auto_approval_limit", "manager_approval_limit"}


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


def extract_labeled_claims(text: str) -> list[tuple[str | None, int]]:
    """답변 텍스트에서 (라벨 또는 None, 숫자) 쌍을 추출.

    "라벨: 숫자원"처럼 콜론이 숫자 바로 앞에 오는 구조가 명확한 claim만 라벨링한다.
    콤마로 나열된 형태나 콜론 없는 자유 서술문은 여기서 라벨을 얻지 못하고 None으로
    남아 grounding_check()의 존재-여부(Level 1/3) 체크로만 검증된다 — 의도된 동작이다.
    """
    claims: list[tuple[str | None, int]] = []
    for m in _LABEL_VALUE_PATTERN.finditer(text):
        label_text, num_str, man = m.groups()
        try:
            value = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if man:
            value *= 10000
        value = int(value)

        value_tail = text[m.end(): m.end() + 20]
        is_range_context = any(h in label_text or h in value_tail for h in _RANGE_HINTS)

        label = None
        for candidate_label, keywords in _LABEL_KEYWORDS:
            if is_range_context and candidate_label in _APPROVAL_LABELS:
                continue
            if any(kw in label_text for kw in keywords):
                label = candidate_label
                break

        claims.append((label, value))
    return claims


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


def _walk_and_collect(obj: Any, evidence: dict[str, set[int]]) -> None:
    """ToolMessage에서 파싱한 JSON을 재귀적으로 훑어 알려진 필드명을 라벨 evidence로 수집.
    전부 도구가 실제로 반환한 값만 다룬다 — evaluator가 새로 계산하는 값은 없다."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _FIELD_LABEL_MAP and isinstance(value, (int, float)) and not isinstance(value, bool):
                evidence[_FIELD_LABEL_MAP[key]].add(int(value))
            _walk_and_collect(value, evidence)
    elif isinstance(obj, list):
        for item in obj:
            _walk_and_collect(item, evidence)


def _derive_typed_evidence(evidence: dict[str, set[int]], welfare_ratios: set[int]) -> dict[str, set[int]]:
    """의미가 명확히 정의된 파생값만 계산한다. 임의의 두 숫자 조합이 아니라
    특정 라벨 쌍에 대해서만, 명시된 관계식으로만 허용한다.

    budget_shortfall = 구매금액 - remaining_budget (구매금액 > remaining_budget일 때)

    구매금액 후보는 기본적으로 total_price다. unit_price는 quantity 증거가 아예
    없거나(purchase_request/purchase_register가 호출 안 돼 quantity를 관측 못한 경우)
    quantity==1로 관측된 경우에만 보조 후보로 추가한다 — quantity>1인데 unit_price를
    그대로 쓰면 실제 총액과 다른 잘못된 shortfall이 나올 수 있기 때문이다.

    welfare_budget_limit = team_budget × welfare_ratio(%) / 100
    team_budget(예: 배정 예산 8,000,000)는 budget_check 결과에서, welfare_ratio(예: 10)는
    policy_search 결과 원문 중 '복지/간식비' 문맥에서만 각각 실제로 관측됐을 때만 계산한다.
    둘 다 관측되지 않으면 아무 값도 만들지 않는다 — pool 안 임의의 숫자와 임의의 퍼센트를
    조합하는 게 아니라, 이 두 특정 라벨의 조합만 허용한다.
    """
    derived: dict[str, set[int]] = {"budget_shortfall": set(), "welfare_budget_limit": set()}
    price_like = set(evidence.get("total_price", set()))

    quantities = evidence.get("quantity", set())
    if not quantities or quantities == {1}:
        price_like |= evidence.get("unit_price", set())

    for price in price_like:
        for remaining in evidence.get("remaining_budget", set()):
            if price > remaining:
                derived["budget_shortfall"].add(price - remaining)

    for team_budget in evidence.get("team_budget", set()):
        for ratio in welfare_ratios:
            derived["welfare_budget_limit"].add(team_budget * ratio // 100)

    return derived


def build_labeled_evidence(messages: list[Any]) -> dict[str, set[int]]:
    """세션에서 실제로 관측된 도구 결과 JSON + 명시적 파생값(Level 2)으로
    라벨별 evidence 집합을 구성.

    policy_rules.yaml의 정책 상수는 여기 포함하지 않는다 — 그건 '시스템 어딘가의
    진실'이지 '에이전트가 이번 세션에서 관측한 근거'가 아니기 때문이다.
    """
    evidence: dict[str, set[int]] = {label: set() for label, _ in _LABEL_KEYWORDS}
    evidence.setdefault("quantity", set())

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        _walk_and_collect(data, evidence)

    welfare_ratios = _extract_welfare_ratio(get_pool_text(messages))

    for label, values in _derive_typed_evidence(evidence, welfare_ratios).items():
        evidence.setdefault(label, set())
        evidence[label] |= values

    return evidence


def grounding_check(messages: list[Any]) -> dict:
    """단일 시나리오에 대한 grounding_check 평가 (v3).

    Returns:
        {"passed": bool, "detail": str, "claimed_numbers": list[int],
         "ungrounded_numbers": list[int], "label_mismatches": list[dict]}
    """
    final_answer = get_final_answer(messages)
    if not final_answer:
        return {
            "passed": True,
            "detail": "최종 답변 없음 (검증 대상 숫자 없음)",
            "claimed_numbers": [],
            "ungrounded_numbers": [],
            "label_mismatches": [],
        }

    claimed = extract_currency_numbers(final_answer)
    if not claimed:
        return {
            "passed": True,
            "detail": "최종 답변에 금액 언급 없음",
            "claimed_numbers": [],
            "ungrounded_numbers": [],
            "label_mismatches": [],
        }

    pool_text = get_pool_text(messages)
    labeled_evidence = build_labeled_evidence(messages)
    general_pool = extract_currency_numbers(pool_text) | extract_bare_numbers(pool_text)
    for label, values in labeled_evidence.items():
        if label == "quantity":
            continue  # 수량은 금액이 아니므로 존재-확인 풀에는 안 섞는다
        general_pool |= values

    ungrounded = sorted(n for n in claimed if n not in general_pool)

    label_mismatches: list[dict] = []
    for label, value in extract_labeled_claims(final_answer):
        if label is None:
            continue
        label_evidence = labeled_evidence.get(label, set())
        if not label_evidence:
            continue  # 이 라벨에 대한 근거를 세션에서 아예 못 모았으면 판단 보류
        if value not in label_evidence:
            label_mismatches.append({
                "label": label,
                "claimed_value": value,
                "expected_any_of": sorted(label_evidence),
            })

    passed = not ungrounded and not label_mismatches

    if passed:
        detail = f"답변에 언급된 금액 {sorted(claimed)} 모두 관측되었거나 명시적 규칙으로 도출 가능한 근거에서 확인됨"
    else:
        parts = []
        if ungrounded:
            parts.append(f"근거 없이 언급된 금액: {ungrounded}")
        if label_mismatches:
            mismatch_desc = [
                f"'{m['label']}'로 언급된 {m['claimed_value']:,}원 (해당 라벨의 실제 근거값: {m['expected_any_of']})"
                for m in label_mismatches
            ]
            parts.append(f"라벨 불일치: {', '.join(mismatch_desc)}")
        detail = " / ".join(parts)

    return {
        "passed": passed,
        "detail": detail,
        "claimed_numbers": sorted(claimed),
        "ungrounded_numbers": ungrounded,
        "label_mismatches": label_mismatches,
    }


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    def tm(content: dict, name: str, tool_call_id: str = "1") -> ToolMessage:
        return ToolMessage(content=json.dumps(content, ensure_ascii=False), name=name, tool_call_id=tool_call_id)

    # 케이스 1: 정상 - 콜론 없는 자유 서술이라 라벨링 시도 안 하고 존재 여부만 확인 -> pass
    print("케이스1 (pass 기대, 자유서술 존재-확인):", grounding_check([
        HumanMessage(content="마우스 사도 돼?"),
        AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {}, "id": "1"}]),
        tm({"results": [{"text": "구매 금액이 50만원 미만인 경우 자동승인됩니다."}]}, "policy_search"),
        AIMessage(content="9만9천원이라 50만원 미만 기준에 해당해서 자동승인 대상입니다."),
    ]))

    # 케이스 2: 조작 - 어떤 도구 결과에도 없는 숫자 -> fail
    print("케이스2 (fail 기대, ungrounded):", grounding_check([
        HumanMessage(content="노트북 사도 돼?"),
        AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {}, "id": "1"}]),
        tm({"results": [{"text": "구매 금액이 50만원 미만인 경우 자동승인됩니다."}]}, "policy_search"),
        AIMessage(content="위원회 승인 기준은 300만원이라 이 이하는 바로 승인됩니다."),
    ]))

    # 케이스 3: 라벨 불일치 (콜론 구조) - 실제 잔여예산 950,000원인데 총액 300,000원을
    # "잔여 예산:"이라고 잘못 라벨링 -> fail
    print("케이스3 (fail 기대, label_mismatches):", grounding_check([
        HumanMessage(content="인사팀에서 모니터 하나 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "1"}]),
        tm({"team_name": "인사팀", "allocated": 5000000, "spent": 4050000, "remaining": 950000}, "budget_check"),
        AIMessage(content="", tool_calls=[{"name": "purchase_request", "args": {}, "id": "2"}]),
        tm({"request_id": "REQ-0001", "unit_price": 300000, "total_price": 300000, "quantity": 1, "created": True}, "purchase_request", "2"),
        AIMessage(content="인사팀 잔여 예산: 300,000원"),
    ]))

    # 케이스 4: Level 1 직접 근거 - purchase_register의 budget_after.remaining 그대로 인용 -> pass
    print("케이스4 (pass 기대, Level 1 직접 근거):", grounding_check([
        HumanMessage(content="이 구매 등록해줘"),
        AIMessage(content="", tool_calls=[{"name": "purchase_register", "args": {}, "id": "1"}]),
        tm({"request_id": "REQ-0001", "total_price": 300000, "registered": True,
            "budget_after": {"allocated": 1000000, "spent": 300000, "remaining": 700000}}, "purchase_register"),
        AIMessage(content="구매 등록이 완료됐고, 구매 후 잔액은 70만원입니다."),
    ]))

    # 케이스 5: Level 2 typed derivation - purchase_request가 호출 안 돼 quantity 증거가 없는
    # 상태에서 unit_price로 shortfall 계산 -> pass (quantity 미관측이라 unit_price 허용됨)
    print("케이스5 (pass 기대, Level 2 - quantity 미관측 시 unit_price 허용):", grounding_check([
        HumanMessage(content="인사팀에서 마우스 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "1"}]),
        tm({"team_name": "인사팀", "allocated": 5000000, "spent": 4950000, "remaining": 50000}, "budget_check"),
        AIMessage(content="", tool_calls=[{"name": "product_search", "args": {}, "id": "2"}]),
        tm({"results": [{"product_id": "P006", "name": "로지텍 MX Master 3S", "price": 99000}]}, "product_search", "2"),
        AIMessage(content="상품: 로지텍 MX Master 3S\n가격: 99,000원\n인사팀 잔여 예산: 50,000원\n부족 금액: 49,000원"),
    ]))

    # 케이스 6: quantity>1이 명시적으로 관측된 경우 unit_price를 shortfall에 쓰면 안 됨 -> 이 경우
    # 에이전트가 (틀린) unit_price 기반 shortfall을 주장하면 fail 해야 함
    print("케이스6 (fail 기대, quantity>1일 때 unit_price 기반 shortfall은 근거 아님):", grounding_check([
        HumanMessage(content="재무팀에서 A4 복사용지 20박스 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "1"}]),
        tm({"team_name": "재무팀", "allocated": 1000000, "spent": 700000, "remaining": 300000}, "budget_check"),
        AIMessage(content="", tool_calls=[{"name": "purchase_request", "args": {}, "id": "2"}]),
        tm({"request_id": "REQ-0001", "unit_price": 25000, "total_price": 500000, "quantity": 20, "created": True}, "purchase_request", "2"),
        # 진짜 shortfall은 500000-300000=200000인데, 에이전트가 단가 기준(25000-300000<0, 말이 안 되지만
        # 예시로 임의의 오답 12345를 부족액이라 주장했다고 가정 -> 근거 없음
        AIMessage(content="총액: 500,000원\n잔여 예산: 300,000원\n부족 금액: 12,345원"),
    ]))

    # 케이스 7: 콤마 나열형 - 콜론 구조가 아니라서 라벨링 자체를 시도 안 하고 존재-확인으로 폴백,
    # 세 값 다 tool JSON에 실제로 있으므로 -> pass (known limitation: 의미 검증은 안 하지만 오탐도 없음)
    print("케이스7 (pass 기대, 콤마 나열형은 라벨링 보류 + 존재확인 통과):", grounding_check([
        HumanMessage(content="인사팀 예산 얼마야"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "1"}]),
        tm({"team_name": "인사팀", "allocated": 8000000, "spent": 2100000, "remaining": 5900000}, "budget_check"),
        AIMessage(content="인사팀 예산: 배정 800만 원, 사용 210만 원, 잔여 590만 원"),
    ]))

    # 케이스 8: 키워드는 있는데 바로 옆에 숫자가 없는 프리즈 문장(S06 재현) - 라벨링 시도 안 하고
    # 존재-확인으로 폴백, 50만원은 실제로 세션에 있으므로 -> pass (v2 window 방식에서 실패했던 케이스)
    print("케이스8 (pass 기대, '잔여'가 근처 숫자 없이 쓰인 문장):", grounding_check([
        HumanMessage(content="인사팀에서 마우스 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {}, "id": "1"}]),
        tm({"results": [{"text": "구매 금액이 50만원 미만인 경우 자동승인됩니다."}]}, "policy_search"),
        AIMessage(content="상품은 재고가 있고 50만 원 미만이라 자동 승인 대상이지만, 인사팀 잔여 예산이 부족하여 현재는 구매 등록을 진행할 수 없습니다."),
    ]))

    # 케이스 9: 같은 줄 다중 클레임(콜론 구조) - 거리 기반이 아니라 콜론 앵커링이라 정확히 분리 -> pass
    result9 = grounding_check([
        HumanMessage(content="A4 복사용지 20박스 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "purchase_request", "args": {}, "id": "1"}]),
        tm({"request_id": "REQ-0001", "unit_price": 25000, "total_price": 500000, "quantity": 20, "created": True}, "purchase_request"),
        AIMessage(content="단가: 25,000원 / 총액: 500,000원 입니다."),
    ])
    print("케이스9 (pass 기대, 한 줄 다중 클레임):", result9)
    print("  -> extract_labeled_claims:", extract_labeled_claims("단가: 25,000원 / 총액: 500,000원 입니다."))

    # 케이스 9b: 키워드 충돌 회귀 테스트 - "부족 금액:"이 total_price의 "금액" 키워드에
    # 잘못 걸려 budget_shortfall 대신 total_price로 오분류되는 버그가 실제로 있었음
    # (total_price 키워드에 범용 "금액"을 넣었던 v3 초안에서 발견, 제거로 수정).
    print("케이스9b (부족 금액 -> budget_shortfall로 정확히 라벨링돼야 함):",
          extract_labeled_claims("부족 금액: 12,345원"))

    # 케이스 10: 알려진 한계 - entity 구분 미지원 (인사팀 잔여예산을 물었는데 AI개발팀 숫자로 답함)
    print("케이스10 (알려진 한계로 pass 기대, entity 구분 미지원):", grounding_check([
        HumanMessage(content="인사팀이랑 AI개발팀 예산 각각 얼마 남았어?"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "1"}]),
        tm({"team_name": "인사팀", "allocated": 5000000, "spent": 4050000, "remaining": 950000}, "budget_check"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "2"}]),
        tm({"team_name": "AI개발팀", "allocated": 15000000, "spent": 3450000, "remaining": 11550000}, "budget_check", "2"),
        AIMessage(content="인사팀 잔여 예산: 11,550,000원"),
    ]))

    # 케이스 11: welfare_budget_limit typed derivation - S04 실제 재현. team_budget(800만원)은
    # budget_check로, 10%는 policy_search 결과 원문의 '복지/간식비' 문맥에서 각각 관측됐고,
    # 에이전트가 그 곱(80만원)을 콜론 없이 자유서술로 언급 -> 존재확인 풀에 파생값이 들어가서 pass
    # (v3 초판에서는 이 80만원을 ungrounded로 오분류했던 실제 버그의 재현 테스트)
    print("케이스11 (pass 기대, welfare_budget_limit 파생 - S04 재현):", grounding_check([
        HumanMessage(content="인사팀에서 팀 회식용으로 와인 선물세트 하나 사고 싶어"),
        AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {}, "id": "1"}]),
        tm({"results": [{"text": "복지/간식비: 팀 분기 예산의 10% 이내에서만 집행 가능"}]}, "policy_search"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "2"}]),
        tm({"team_name": "인사팀", "allocated": 8000000, "spent": 2100000, "remaining": 5900000}, "budget_check", "2"),
        AIMessage(content="인사팀 예산: 잔여 590만 원\n복지·간식비는 팀 분기 예산의 10% 이내(80만 원)에서 집행 가능합니다."),
    ]))

    # 케이스 12: welfare_budget_limit 라벨 불일치 - 콜론 구조로 잘못된 한도(90만원)를 주장하면
    # 실제 파생값(80만원)과 달라서 fail 해야 함
    print("케이스12 (fail 기대, welfare_budget_limit 콜론 구조 오답):", grounding_check([
        HumanMessage(content="인사팀 복지비 한도가 얼마야"),
        AIMessage(content="", tool_calls=[{"name": "policy_search", "args": {}, "id": "1"}]),
        tm({"results": [{"text": "복지/간식비: 팀 분기 예산의 10% 이내에서만 집행 가능"}]}, "policy_search"),
        AIMessage(content="", tool_calls=[{"name": "budget_check", "args": {}, "id": "2"}]),
        tm({"team_name": "인사팀", "allocated": 8000000, "spent": 2100000, "remaining": 5900000}, "budget_check", "2"),
        AIMessage(content="복지 한도: 900,000원"),
    ]))