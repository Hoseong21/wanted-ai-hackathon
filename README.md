# AfA(Agent for Aengt) — 검증 가능한 Enterprise AI Agent

RAG 기반 사내 구매 업무 에이전트에 Grounding Check / Tool Call Check / Policy Check
검증 계층을 추가하여, "정답을 내는 에이전트"가 아니라 "검증 가능하게 정답을 내는 에이전트"를
지향하는 프로젝트입니다. (원티드 AI Championship 2026 출품작)

## 프로젝트 구조

- `data/` — 가짜 사내 구매규정 문서, mock 제품 카탈로그, 예산 DB 초기화 스크립트
- `src/rag/` — 문서 적재(ingest) 및 검색(retriever)
- `src/tools/` — 에이전트가 호출하는 도구 (제품 검색, 예산 확인, 구매요청서 작성, 구매 등록)
- `src/agent/` — LangGraph 기반 Task Planner + Agent 그래프
- `src/evaluators/` — Grounding / Tool Call / Policy 검증 계층
- `src/logging/` — 실행 로그
- `app/` — Streamlit 기반 Human Approval UI
- `experiments/` — 정상/실패유도 시나리오 및 baseline vs agent vs agent+verification 비교 실험

## 실행 방법

TODO: 환경 세팅 및 실행 커맨드 정리 예정

## 아키텍처

TODO: 아키텍처 다이어그램, 설계 이유, 파이프라인 플로우차트, ERD (MVP 완성 후 정리)
