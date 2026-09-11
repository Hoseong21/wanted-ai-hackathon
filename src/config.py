"""프로젝트 전역 설정."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# data
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
PRODUCTS_PATH = DATA_DIR / "products.json"
BUDGET_DB_PATH = Path(os.getenv("BUDGET_DB_PATH", str(DATA_DIR / "budget.db")))

# rag
VECTORSTORE_DIR = BASE_DIR / "src" / "rag" / "vectorstore"
COLLECTION_NAME = "purchase_policy"

# llm
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6-luna")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
