"""budget.db 초기화 CLI 진입점. 실제 로직은 src/db/init_budget_db.py에 있다.

실행 (repo 루트에서):
    python3 -m data.init_budget_db
"""
from src.db.init_budget_db import init_budget_db

if __name__ == "__main__":
    init_budget_db()