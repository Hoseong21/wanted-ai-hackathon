"""정책 문서 검색 도구 (mock, judgment-free).

purchase_policy 벡터DB(Chroma + KURE-v1 1단계, KURE-v2 2단계 rerank)에서 질의와 관련된 정책 문서
조각을 찾아 사실(fact)만 반환한다. 이 정책이 지금 상황에 적용되는지, 위반인지 등의 판단은 하지
않는다 — 그건 에이전트와 evaluators/outcome_check.py의 몫.

에이전트가 "필요하다고 판단할 때만" 호출하는 5번째 툴이라는 점이 중요하다: 자동으로 매 질문마다
실행되지 않는다. 에이전트가 이 툴을 호출했는지/안 했는지 자체가 grounding_check의 평가 대상이 된다.
"""

from __future__ import annotations

from src.rag.retriever import search_and_rerank


def policy_search(query: str, top_k: int = 3) -> dict:
    """사내 구매 정책 문서에서 질의와 관련된 조각을 검색한다.

    Args:
        query: 검색할 정책 관련 질문/키워드 (예: "50만원 이상 구매 승인")
        top_k: 반환할 최대 결과 수 (기본 3)

    Returns:
        성공 시: {"results": [{"text", "source", "chunk_index", "rerank_score"}, ...], "count": int}
        실패 시: {"error": True, "reason": str, "message": str, "hint": str}
    """
    try:
        hits = search_and_rerank(query, k=top_k)
    except Exception as exc:  # noqa: BLE001 - 벡터DB 미구축 등 다양한 실패를 사용자에게 알려야 함
        return {
            "error": True,
            "reason": "POLICY_SEARCH_FAILED",
            "message": f"정책 문서 검색 중 오류가 발생했습니다: {exc}",
            "hint": "python3 -m src.rag.ingest로 벡터DB가 구축돼있는지 확인하세요.",
        }

    results = [
        {
            "text": hit["text"],
            "source": hit["source"],
            "chunk_index": hit["chunk_index"],
            "rerank_score": hit["rerank_score"],
        }
        for hit in hits
    ]
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import json
    import sys

    query = " ".join(sys.argv[1:]) or "50만원 이상 구매할 때 승인은 누구한테 받아야 해?"
    print(json.dumps(policy_search(query), ensure_ascii=False, indent=2))