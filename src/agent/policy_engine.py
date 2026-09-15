"""런타임 정책 판정기 — 라벨 없이 동작하는 verify 게이트의 핵심.

Evaluator(오프라인, 정답 라벨 필요)와 다르게 이 모듈은 정답을 보지 않고
policy_rules.yaml + 실제 DB/카탈로그 값만으로 즉석에서 ALLOW/REQUIRE_APPROVAL/BLOCK을 판정한다.

핵심 원칙: LLM이 넘긴 인자를 믿지 않는다. purchase_register는 애초에 request_id만 받으므로,
금액/팀/상품 정보는 전부 DB에서 직접 재조회한다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from src import config

_RULES_PATH = Path(__file__).parent / "policy_rules.yaml"


def _load_rules() -> dict:
    return yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))


def _load_product(product_id: str) -> dict | None:
    products = json.loads(config.PRODUCTS_PATH.read_text(encoding="utf-8"))
    return next((p for p in products if p["id"] == product_id), None)


def _get_purchase_request(request_id: int) -> dict | None:
    conn = sqlite3.connect(config.BUDGET_DB_PATH)
    try:
        row = conn.execute(
            "SELECT team_name, product_id, quantity, total_price, registered_at "
            "FROM purchase_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    team_name, product_id, quantity, total_price, registered_at = row
    return {
        "team_name": team_name,
        "product_id": product_id,
        "quantity": quantity,
        "total_price": total_price,
        "already_registered": registered_at is not None,
    }


def _get_remaining_budget(team_name: str) -> int | None:
    conn = sqlite3.connect(config.BUDGET_DB_PATH)
    try:
        row = conn.execute(
            "SELECT allocated_budget, spent_amount FROM budget "
            "JOIN teams ON budget.team_id = teams.team_id "
            "WHERE teams.team_name = ? ORDER BY year DESC, quarter DESC LIMIT 1",
            (team_name,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    allocated, spent = row
    return allocated - spent


def evaluate_purchase_register(request_id: str) -> dict:
    """purchase_register 실행 여부를 게이트에서 판정.

    Returns: {"decision": "ALLOW"|"REQUIRE_APPROVAL"|"BLOCK", "reason": str, "facts": dict}
    """
    rules = _load_rules()

    if not request_id.startswith("REQ-"):
        return {"decision": "BLOCK", "reason": f"잘못된 request_id 형식: {request_id}", "facts": {}}
    try:
        numeric_id = int(request_id[4:])
    except ValueError:
        return {"decision": "BLOCK", "reason": f"잘못된 request_id 형식: {request_id}", "facts": {}}

    req = _get_purchase_request(numeric_id)
    if req is None:
        return {"decision": "BLOCK", "reason": f"request_id '{request_id}' 없음", "facts": {}}
    if req["already_registered"]:
        return {"decision": "BLOCK", "reason": "이미 등록된 요청 (중복 등록)", "facts": req}

    product = _load_product(req["product_id"])
    if product is None:
        return {"decision": "BLOCK", "reason": f"product_id '{req['product_id']}' 카탈로그에 없음", "facts": req}

    remaining = _get_remaining_budget(req["team_name"])
    if remaining is None:
        return {"decision": "BLOCK", "reason": f"'{req['team_name']}' 예산 데이터 없음", "facts": req}

    total_price = req["total_price"]
    category = product["category"]
    is_restricted = any(kw in product["name"] for kw in rules.get("restricted_keywords", []))
    facts = {**req, "category": category, "remaining_budget_before": remaining, "is_restricted": is_restricted}

    # 1. 금지 품목 — 예산/승인 여부와 무관하게 무조건 차단
    if category in rules["forbidden_categories"]:
        return {"decision": "BLOCK", "reason": f"금지 품목 카테고리('{category}')", "facts": facts}

    # 2. 예산 부족 — 승인 여부와 무관하게 차단
    if total_price > remaining:
        return {
            "decision": "BLOCK",
            "reason": f"예산 부족 (요청 {total_price:,}원 > 잔여 {remaining:,}원)",
            "facts": facts,
        }

    auto_limit = rules["approval"]["auto_approval_limit"]
    manager_limit = rules["approval"]["manager_approval_limit"]

    # 3. 금액 구간
    if total_price >= manager_limit:
        return {"decision": "REQUIRE_APPROVAL", "reason": f"{total_price:,}원, 부서장 승인 구간(≥{manager_limit:,}원)", "facts": facts}

    # 4. 리퍼비시/중고 — 금액과 무관하게 승인 필요
    if is_restricted:
        return {"decision": "REQUIRE_APPROVAL", "reason": "리퍼비시/중고 제품, 팀장 사전 승인 필요", "facts": facts}

    if total_price >= auto_limit:
        return {"decision": "REQUIRE_APPROVAL", "reason": f"{total_price:,}원, 팀장 승인 구간(≥{auto_limit:,}원)", "facts": facts}

    return {"decision": "ALLOW", "reason": f"{total_price:,}원, 자동승인 구간(<{auto_limit:,}원)", "facts": facts}


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate_purchase_register(sys.argv[1]), ensure_ascii=False, indent=2, default=str))