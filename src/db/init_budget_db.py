"""SQLite 예산 DB 스키마 생성 및 시드 데이터 삽입 로직.

CLI 진입점은 data/init_budget_db.py에 얇게 남아 있고, 실제 로직은 여기 하나로 모아서
experiments/run_comparison.py와 Streamlit 앱이 공통으로 import해서 쓴다.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from src import config

TEAMS = ["AI개발팀", "마케팅팀", "인사팀", "재무팀"]

# (team_name, year, quarter, allocated_budget, spent_amount)
BUDGET_SEED = [
    ("AI개발팀", 2026, 3, 20_000_000, 8_450_000),
    ("마케팅팀", 2026, 3, 15_000_000, 12_300_000),
    ("인사팀", 2026, 3, 8_000_000, 2_100_000),
    ("재무팀", 2026, 3, 6_000_000, 1_050_000),
]


def init_budget_db(db_path: Path = config.BUDGET_DB_PATH, reset: bool = True) -> None:
    """budget.db에 teams / budget / purchase_requests 테이블을 만들고 mock 데이터를 채운다."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if reset:
        cur.execute("DROP TABLE IF EXISTS purchase_requests")
        cur.execute("DROP TABLE IF EXISTS budget")
        cur.execute("DROP TABLE IF EXISTS teams")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS budget (
            budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL,
            allocated_budget INTEGER NOT NULL,
            spent_amount INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (team_id) REFERENCES teams (team_id),
            UNIQUE (team_id, year, quarter)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS purchase_requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            requester TEXT NOT NULL,
            unit_price INTEGER NOT NULL,
            total_price INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            registered_at TEXT
        )
        """
    )

    for team_name in TEAMS:
        cur.execute("INSERT OR IGNORE INTO teams (team_name) VALUES (?)", (team_name,))

    for team_name, year, quarter, allocated, spent in BUDGET_SEED:
        cur.execute("SELECT team_id FROM teams WHERE team_name = ?", (team_name,))
        team_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT OR REPLACE INTO budget
                (team_id, year, quarter, allocated_budget, spent_amount)
            VALUES (?, ?, ?, ?, ?)
            """,
            (team_id, year, quarter, allocated, spent),
        )

    conn.commit()
    conn.close()
    print(f"budget db created at {db_path}")


def create_temp_budget_db() -> Path:
    """독립된 임시 budget DB를 만들어 그 경로를 반환한다.

    실험 하네스(REPEATS 반복 실행)와 Streamlit 세션이 공통으로 사용한다.
    파일 정리는 호출자 책임 — 실험 코드는 with 블록 종료 시 unlink하고,
    Streamlit 세션은 의도적으로 정리하지 않는다(재배포/재부팅에 맡김).
    """
    fd, path_str = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    tmp_path = Path(path_str)
    init_budget_db(db_path=tmp_path, reset=True)
    return tmp_path


if __name__ == "__main__":
    init_budget_db()