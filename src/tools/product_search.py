"""상품 검색 도구 (mock, judgment-free).

data/products.json의 상품 목록에서 조건에 맞는 상품을 찾아 사실(fact)만 반환한다.
"구매 가능/불가능" 같은 판단은 하지 않는다 — 그건 evaluators/outcome_check.py의 몫.
"""

from __future__ import annotations

import json

from src.config import PRODUCTS_PATH

VALID_CATEGORIES = {
    "노트북", "모니터", "주변기기", "소프트웨어", "사무용품", "기타", "금지품목", "복지",
}


def _load_products() -> list[dict]:
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def product_search(query: str, category: str | None = None, max_price: int | None = None) -> dict:
    """상품명/키워드로 상품을 검색한다.

    Args:
        query: 검색할 상품명 또는 키워드 (부분 일치, 대소문자 구분 안 함)
        category: 카테고리 필터. None이면 전체 카테고리 검색
        max_price: 최대 가격(원) 필터. None이면 제한 없음

    Returns:
        성공 시: {"results": [{"product_id", "name", "category", "price"}, ...], "count": int}
        실패 시: {"error": True, "reason": str, "message": str, "hint": str}
    """
    if category is not None and category not in VALID_CATEGORIES:
        return {
            "error": True,
            "reason": "INVALID_CATEGORY",
            "message": f"'{category}'는 유효한 카테고리가 아닙니다.",
            "hint": f"다음 중 하나를 사용하세요: {', '.join(sorted(VALID_CATEGORIES))}",
        }

    products = _load_products()
    query_lower = query.lower()

    results = []
    for p in products:
        if query_lower not in p["name"].lower():
            continue
        if category is not None and p["category"] != category:
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        results.append({
            "product_id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "spec": p["spec"],
            "in_stock": p["in_stock"],
        })

    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "노트북"
    result = product_search(query)
    print(f"query: {query}")
    print(f"count: {result['count']}")
    for r in result["results"]:
        print(f"  [{r['product_id']}] {r['name']} ({r['category']}) - {r['price']:,}원")