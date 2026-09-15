"""RAG baseline: 툴콜 루프(ReAct) 없이, 정책 문서 검색(policy_search와 동일한 RAG
파이프라인) 결과만 컨텍스트로 주고 LLM이 한 번에 답변한다. 액션(구매 요청/등록 등)은
절대 실행하지 않는다.

3조건 비교 실험의 (a) 조건: "RAG는 있지만 에이전트 아키텍처(툴콜 루프)가 없으면 얼마나
부족한가"를 보여주기 위한 비교군. product_search/budget_check 같은 구조화된 DB/API
조회는 의도적으로 제외한다 — "RAG baseline"이라는 이름 그대로 검색증강생성만 반영하고,
그 외 도구 접근은 agent(b)/agent+verification(c) 조건에서만 허용되는 차이점이기 때문이다.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from src.rag.retriever import search_and_rerank

BASELINE_SYSTEM_PROMPT = (
    "당신은 사내 구매 문의에 답변하는 어시스턴트입니다. 아래 제공된 사내 정책 문서 검색 "
    "결과만을 근거로 답변하세요. 문서에 없는 내용은 추측하지 말고 모른다고 답하세요. "
    "당신은 어떤 구매 요청도 실제로 접수하거나 등록/승인할 수 없습니다 — 답변은 정보 "
    "제공 목적일 뿐입니다. 응답은 한국어로 하세요."
)

_baseline_llm = None


def _get_baseline_llm() -> ChatOpenAI:
    global _baseline_llm
    if _baseline_llm is None:
        _baseline_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            reasoning_effort="none",
        )
    return _baseline_llm


def run_baseline(user_query: str, top_k: int = 3) -> dict:
    """RAG baseline 실행.

    Returns:
        {"answer": str, "retrieved_context": list[dict]}
        retrieved_context 항목: {"text", "source", "chunk_index", "rerank_score"}
    """
    hits = search_and_rerank(user_query, k=top_k)

    context_text = "\n\n".join(
        f"[출처: {h['source']}]\n{h['text']}" for h in hits
    ) or "(관련 정책 문서를 찾지 못했습니다)"

    prompt = f"[검색된 정책 문서]\n{context_text}\n\n[질문]\n{user_query}"

    response = _get_baseline_llm().invoke(
        [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return {
        "answer": response.content,
        "retrieved_context": hits,
    }


if __name__ == "__main__":
    import json
    import sys

    query = " ".join(sys.argv[1:]) or "인사팀에서 로지텍 MX Master 3S 마우스 하나 사려고 해. 요청자는 김철수야. 가능하면 등록까지 진행해줘."
    result = run_baseline(query)
    print("답변:\n", result["answer"])
    print("\n검색된 문서:")
    print(json.dumps(result["retrieved_context"], ensure_ascii=False, indent=2))