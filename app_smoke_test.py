"""
배포 스모크 테스트 앱.
실제 RAG(ingest/search_and_rerank)와 LLM(_get_llm) 경로를 그대로 호출해서
Streamlit Community Cloud에서 정상 기동되는지, 단계별 소요 시간을 확인한다.

Secrets 설정 (Streamlit Cloud 앱 설정 > Secrets, TOML 형식):
OPENAI_API_KEY = "sk-..."
"""
import os
import time

import streamlit as st

# src.config가 import 시점에 os.getenv로 읽으므로, src 임포트보다 먼저 반영해야 함
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain_core.messages import HumanMessage  # noqa: E402

from src.agent.graph import _get_llm  # noqa: E402
from src.rag.ingest import ingest  # noqa: E402
from src.rag.retriever import search_and_rerank  # noqa: E402

st.title("배포 스모크 테스트")


@st.cache_resource(show_spinner=False)
def _ensure_vectorstore() -> int:
    """프로세스 생존 기간 동안 최초 1회만 실행됨."""
    return ingest()


if st.button("스모크 테스트 실행"):
    results = []

    t0 = time.time()
    try:
        n_chunks = _ensure_vectorstore()
        results.append(("벡터DB ingest (KURE-v1 임베딩)", True, time.time() - t0, f"{n_chunks}개 청크"))
    except Exception as e:
        results.append(("벡터DB ingest (KURE-v1 임베딩)", False, time.time() - t0, str(e)))

    t0 = time.time()
    try:
        hits = search_and_rerank("50만원 이상 구매할 때 승인은 누구한테 받아야 해?", k=3)
        results.append(("RAG 검색+rerank (KURE-v1+KURE-v2)", True, time.time() - t0, f"{len(hits)}건 반환"))
    except Exception as e:
        results.append(("RAG 검색+rerank (KURE-v1+KURE-v2)", False, time.time() - t0, str(e)))

    t0 = time.time()
    try:
        resp = _get_llm().invoke([HumanMessage(content="테스트: 잘 작동하는지 한 문장으로만 답해줘.")])
        results.append(("OpenAI 호출 (_get_llm)", True, time.time() - t0, resp.content[:80]))
    except Exception as e:
        results.append(("OpenAI 호출 (_get_llm)", False, time.time() - t0, str(e)))

    total = sum(r[2] for r in results)
    st.write(f"테스트 실행 소요 시간: {total:.1f}s (첫 실행 시 모델 로딩 시간 포함)")
    for label, ok, elapsed, detail in results:
        icon = "성공" if ok else "실패"
        st.write(f"[{icon}] {label} — {elapsed:.1f}s — {detail}")