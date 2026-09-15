"""구매 등록(집행) 도구 (mock, judgment-free).

접수된 구매 요청(purchase_request)을 실제로 집행해 budget.db의 spent_amount에 반영한다.
이 요청이 정책상 타당한지, 승인돼야 하는지는 판단하지 않는다 — 그건 에이전트와
evaluators/outcome_check.py가 이 툴을 호출하기 전에 이미 판단을 마쳤어야 하는 영역이다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src import config


def _parse_request_id(request_id: str) -> int | None:
    if not request_id.startswith("REQ-"):
        return None
    try:
        return int(request_id[4:])
    except ValueError:
        return None


def purchase_register(request_id: str) -> dict:
    """구매 요청을 집행해 예산에 실제로 반영한다.

    Args:
        request_id: purchase_request로 발급받은 요청 ID (예: "REQ-0001")

    Returns:
        성공 시: {"request_id", "team_name", "product_id", "quantity", "total_price",
                  "registered": True, "budget_after": {"allocated", "spent", "remaining"}}
        실패 시: {"error": True, "reason": str, "message": str, "hint": str}
    """
    numeric_id = _parse_request_id(request_id)
    if numeric_id is None:
        return {
            "error": True,
            "reason": "INVALID_REQUEST_ID",
            "message": f"'{request_id}'는 유효한 request_id 형식이 아닙니다.",
            "hint": "purchase_request가 반환한 'REQ-0001' 형식의 request_id를 사용하세요.",
        }

    conn = sqlite3.connect(config.BUDGET_DB_PATH)
    try:
        row = conn.execute(
            "SELECT team_name, product_id, quantity, total_price, registered_at "
            "FROM purchase_requests WHERE request_id = ?",
            (numeric_id,),
        ).fetchone()

        if row is None:
            return {
                "error": True,
                "reason": "REQUEST_NOT_FOUND",
                "message": f"request_id '{request_id}'를 찾을 수 없습니다.",
                "hint": "purchase_request로 먼저 요청을 접수하세요.",
            }

        team_name, product_id, quantity, total_price, registered_at = row

        if registered_at is not None:
            return {
                "error": True,
                "reason": "ALREADY_REGISTERED",
                "message": f"request_id '{request_id}'는 이미 {registered_at}에 등록됐습니다.",
                "hint": "동일한 request_id로 중복 등록할 수 없습니다.",
            }

        budget_row = conn.execute(
            "SELECT budget_id, allocated_budget, spent_amount FROM budget "
            "JOIN teams ON budget.team_id = teams.team_id "
            "WHERE teams.team_name = ? ORDER BY year DESC, quarter DESC LIMIT 1",
            (team_name,),
        ).fetchone()

        if budget_row is None:
            return {
                "error": True,
                "reason": "BUDGET_NOT_FOUND",
                "message": f"'{team_name}'의 예산 데이터를 찾을 수 없습니다.",
                "hint": "data/init_budget_db.py가 정상적으로 실행됐는지 확인하세요.",
            }

        budget_id, allocated, spent = budget_row
        new_spent = spent + total_price
        registered_at_ts = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "UPDATE budget SET spent_amount = ? WHERE budget_id = ?",
            (new_spent, budget_id),
        )
        conn.execute(
            "UPDATE purchase_requests SET registered_at = ? WHERE request_id = ?",
            (registered_at_ts, numeric_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "request_id": request_id,
        "team_name": team_name,
        "product_id": product_id,
        "quantity": quantity,
        "total_price": total_price,
        "registered": True,
        "budget_after": {
            "allocated": allocated,
            "spent": new_spent,
            "remaining": allocated - new_spent,
        },
    }


if __name__ == "__main__":
    import json
    import sys

    args = sys.argv[1:]
    if len(args) != 1:
        print("사용법: python3 -m src.tools.purchase_register <request_id>")
        sys.exit(1)

    result = purchase_register(args[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))