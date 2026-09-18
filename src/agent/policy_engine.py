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
from datetime import datetime, timedelta, timezone

_RULES_PATH = Path(__file__).parent / "policy_rules.yaml"


def _load_rules() -> dict:
    return yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))


def _load_product(product_id: str) -> dict | None:
    products = json.loads(config.PRODUCTS_PATH.read_text(encoding="utf-8"))
    return next((p for p in products if p["id"] == product_id), None)


def _get_purchase_request(request_id: int, db_path: Path) -> dict | None:
    conn = sqlite3.connect(db_path)
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


def _get_remaining_budget(team_name: str, db_path: Path) -> int | None:
    conn = sqlite3.connect(db_path)
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


def _sum_recent_amount(
    team_name: str, category: str, window_days: int, exclude_request_id: int, db_path: Path
) -> int:
    """최근 window_days일 내, 동일 팀·카테고리로 이미 등록된 구매 금액의 합계."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT product_id, total_price FROM purchase_requests "
            "WHERE team_name = ? AND registered_at IS NOT NULL AND registered_at >= ? AND request_id != ?",
            (team_name, cutoff, exclude_request_id),
        ).fetchall()
    finally:
        conn.close()

    products = {p["id"]: p for p in json.loads(config.PRODUCTS_PATH.read_text(encoding="utf-8"))}
    return sum(
        total_price for product_id, total_price in rows
        if (p := products.get(product_id)) and p["category"] == category
    )


def evaluate_purchase_register(request_id: str, db_path: Path | None = None) -> dict:
    """purchase_register 실행 여부를 게이트에서 판정.

    Args:
        request_id: 검증할 요청 ID
        db_path: 조회할 budget DB 경로. 지정하지 않으면 config.BUDGET_DB_PATH를 사용한다
            (세션/시나리오별로 격리된 DB를 쓸 때 명시적으로 넘긴다).

    Returns: {"decision": "ALLOW"|"REQUIRE_APPROVAL"|"BLOCK", "reason": str, "facts": dict}
    """
    resolved_path = db_path or config.BUDGET_DB_PATH
    rules = _load_rules()

    if not request_id.startswith("REQ-"):
        return {"decision": "BLOCK", "reason": f"요청 ID '{request_id}'의 형식이 올바르지 않습니다.", "facts": {}}
    try:
        numeric_id = int(request_id[4:])
    except ValueError:
        return {"decision": "BLOCK", "reason": f"요청 ID '{request_id}'의 형식이 올바르지 않습니다.", "facts": {}}

    req = _get_purchase_request(numeric_id, resolved_path)
    if req is None:
        return {"decision": "BLOCK", "reason": f"요청 ID '{request_id}'에 해당하는 구매 요청을 찾을 수 없습니다.", "facts": {}}
    if req["already_registered"]:
        return {"decision": "BLOCK", "reason": "이미 등록이 완료된 요청입니다. (중복 등록 시도)", "facts": req}

    product = _load_product(req["product_id"])
    if product is None:
        return {"decision": "BLOCK", "reason": f"제품 ID '{req['product_id']}'는 제품 카탈로그에 존재하지 않습니다.", "facts": req}

    remaining = _get_remaining_budget(req["team_name"], resolved_path)
    if remaining is None:
        return {"decision": "BLOCK", "reason": f"'{req['team_name']}' 팀의 예산 데이터를 찾을 수 없습니다.", "facts": req}

    total_price = req["total_price"]
    category = product["category"]
    is_restricted = any(kw in product["name"] for kw in rules.get("restricted_keywords", []))
    facts = {**req, "category": category, "remaining_budget_before": remaining, "is_restricted": is_restricted}

    # 1. 금지 품목 — 예산/승인 여부와 무관하게 무조건 차단
    if category in rules["forbidden_categories"]:
        return {"decision": "BLOCK", "reason": f"'{category}'는 금지된 품목 카테고리이므로 구매할 수 없습니다.", "facts": facts}

    # 2. 예산 부족 — 승인 여부와 무관하게 차단
    if total_price > remaining:
        return {
            "decision": "BLOCK",
            "reason": f"요청 금액 {total_price:,}원이 팀의 잔여 예산 {remaining:,}원을 초과합니다.",
            "facts": facts,
        }

    auto_limit = rules["approval"]["auto_approval_limit"]
    manager_limit = rules["approval"]["manager_approval_limit"]

    # 3. 금액 구간
    if total_price >= manager_limit:
        return {"decision": "REQUIRE_APPROVAL", "reason": f"요청 금액 {total_price:,}원은 부서장 승인이 필요한 {manager_limit:,}원 이상 구간입니다.", "facts": facts}

    # 4. 리퍼비시/중고 — 금액과 무관하게 승인 필요
    if is_restricted:
        return {"decision": "REQUIRE_APPROVAL", "reason": "리퍼비시/중고 제품은 팀장의 사전 승인이 필요합니다.", "facts": facts}

    if total_price >= auto_limit:
        return {"decision": "REQUIRE_APPROVAL", "reason": f"요청 금액 {total_price:,}원은 팀장 승인이 필요한 {auto_limit:,}원 이상 구간입니다.", "facts": facts}

    # 5. 분할구매 의심 — 이번 건 자체는 자동승인 구간이지만, 최근 N일간 동일 팀·카테고리 누적 금액과
    #    합산하면 자동승인 한도를 넘는 경우. 고액 구매를 여러 건으로 쪼개서 승인을 우회하는 패턴을 잡는다.
    split_rules = rules.get("split_purchase", {})
    if split_rules.get("enabled"):
        recent_total = _sum_recent_amount(
            req["team_name"], category, split_rules["window_days"], exclude_request_id=numeric_id, db_path=resolved_path,
        )
        combined = recent_total + total_price
        if combined >= auto_limit:
            return {
                "decision": "REQUIRE_APPROVAL",
                "reason": (
                    f"최근 {split_rules['window_days']}일간 '{category}' 카테고리 누적 구매액 {recent_total:,}원에 "
                    f"이번 요청 {total_price:,}원을 더하면 {combined:,}원으로, "
                    f"자동 승인 한도 {auto_limit:,}원을 초과하여 분할 구매가 의심됩니다."
                ),
                "facts": {**facts, "recent_category_total": recent_total, "combined_total": combined},
            }
    return {"decision": "ALLOW", "reason": f"요청 금액 {total_price:,}원이 자동 승인 기준인 {auto_limit:,}원 미만입니다.", "facts": facts}


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate_purchase_register(sys.argv[1]), ensure_ascii=False, indent=2, default=str))