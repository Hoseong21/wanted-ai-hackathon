"""팀 예산 조회 도구 (mock, judgment-free).

data/budget.db에서 팀의 예산 현황을 조회해 사실(fact)만 반환한다.
"예산 충분/부족" 같은 판단은 하지 않는다 — 그건 에이전트와 evaluators/outcome_check.py의 몫.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src import config

VALID_TEAMS = {"AI개발팀", "마케팅팀", "인사팀", "재무팀"}


def budget_check(team_name: str, db_path: Path | None = None) -> dict:
    """팀의 예산 현황을 조회한다.

    Args:
        team_name: 예산을 조회할 팀명
        db_path: 조회할 budget DB 경로. 지정하지 않으면 config.BUDGET_DB_PATH를 사용한다
            (세션/시나리오별로 격리된 DB를 쓸 때 명시적으로 넘긴다).

    Returns:
        성공 시: {"team_name", "period", "allocated", "spent", "remaining"}
        실패 시: {"error": True, "reason": str, "message": str, "hint": str}
    """
    if team_name not in VALID_TEAMS:
        return {
            "error": True,
            "reason": "INVALID_TEAM",
            "message": f"'{team_name}'은(는) 유효한 팀명이 아닙니다.",
            "hint": f"다음 중 하나를 사용하세요: {', '.join(sorted(VALID_TEAMS))}",
        }

    resolved_path = db_path or config.BUDGET_DB_PATH
    conn = sqlite3.connect(resolved_path)
    try:
        row = conn.execute(
            "SELECT year, quarter, allocated_budget, spent_amount FROM budget "
            "JOIN teams ON budget.team_id = teams.team_id "
            "WHERE teams.team_name = ?",
            (team_name,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {
            "error": True,
            "reason": "BUDGET_NOT_FOUND",
            "message": f"'{team_name}'의 예산 데이터를 찾을 수 없습니다.",
            "hint": "data/init_budget_db.py가 정상적으로 실행됐는지 확인하세요.",
        }

    year, quarter, allocated, spent = row
    return {
        "team_name": team_name,
        "period": f"{year}-Q{quarter}",
        "allocated": allocated,
        "spent": spent,
        "remaining": allocated - spent,
    }


if __name__ == "__main__":
    import json
    import sys

    team_name = " ".join(sys.argv[1:]) or "마케팅팀"
    print(json.dumps(budget_check(team_name), ensure_ascii=False, indent=2))