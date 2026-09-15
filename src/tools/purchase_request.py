"""구매 요청 등록 도구 (mock, judgment-free).

팀/상품/수량 정보를 받아 구매 요청 기록을 budget.db의 purchase_requests 테이블에 남긴다.
승인 필요 여부, 정책 위반 여부 등 어떤 판단도 하지 않는다 — 그저 요청이 접수됐다는 사실만 기록한다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src import config
from src.tools.budget_check import VALID_TEAMS
from src.tools.product_search import _load_products


def purchase_request(team_name: str, product_id: str, quantity: int, requester: str) -> dict:
    """구매 요청을 접수하고 기록한다.

    Args:
        team_name: 요청 팀명
        product_id: product_search로 조회한 product_id
        quantity: 수량 (1 이상)
        requester: 요청자 이름 또는 사번

    Returns:
        성공 시: {"request_id", "team_name", "product_id", "quantity", "unit_price", "total_price", "created": True}
        실패 시: {"error": True, "reason": str, "message": str, "hint": str}
    """
    if team_name not in VALID_TEAMS:
        return {
            "error": True,
            "reason": "INVALID_TEAM",
            "message": f"'{team_name}'은(는) 유효한 팀명이 아닙니다.",
            "hint": f"다음 중 하나를 사용하세요: {', '.join(sorted(VALID_TEAMS))}",
        }

    if quantity < 1:
        return {
            "error": True,
            "reason": "INVALID_QUANTITY",
            "message": f"quantity는 1 이상이어야 합니다 (받은 값: {quantity}).",
            "hint": "quantity를 1 이상의 정수로 지정하세요.",
        }

    products = {p["id"]: p for p in _load_products()}
    product = products.get(product_id)
    if product is None:
        return {
            "error": True,
            "reason": "PRODUCT_NOT_FOUND",
            "message": f"product_id '{product_id}'를 찾을 수 없습니다.",
            "hint": "product_search로 유효한 product_id를 먼저 조회하세요.",
        }

    unit_price = product["price"]
    total_price = unit_price * quantity
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(config.BUDGET_DB_PATH)
    try:
        cur = conn.execute(
            """
            INSERT INTO purchase_requests
                (team_name, product_id, quantity, requester, unit_price, total_price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (team_name, product_id, quantity, requester, unit_price, total_price, created_at),
        )
        request_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return {
        "request_id": f"REQ-{request_id:04d}",
        "team_name": team_name,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "created": True,
    }


if __name__ == "__main__":
    import json
    import sys

    args = sys.argv[1:]
    if len(args) != 4:
        print("사용법: python3 -m src.tools.purchase_request <team_name> <product_id> <quantity> <requester>")
        sys.exit(1)

    team_name, product_id, quantity, requester = args
    result = purchase_request(team_name, product_id, int(quantity), requester)
    print(json.dumps(result, ensure_ascii=False, indent=2))