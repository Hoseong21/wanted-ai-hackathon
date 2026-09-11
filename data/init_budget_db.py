"""SQLite 예산 DB 생성 스크립트.

data/budget.db 에 teams / budget 테이블을 만들고 mock 데이터를 삽입한다.
budget_check 툴이 이 DB를 읽어서 팀별 잔여 예산을 조회한다.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "budget.db"

TEAMS = ["AI개발팀", "마케팅팀", "인사팀", "재무팀"]

# (team_name, year, quarter, allocated_budget, spent_amount)
BUDGET_SEED = [
    ("AI개발팀", 2026, 3, 20_000_000, 8_450_000),
    ("마케팅팀", 2026, 3, 15_000_000, 12_300_000),
    ("인사팀", 2026, 3, 8_000_000, 2_100_000),
    ("재무팀", 2026, 3, 6_000_000, 1_050_000),
]


def init_db(db_path: Path = DB_PATH, reset: bool = True) -> None:
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


if __name__ == "__main__":
    init_db()
