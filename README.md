# AfA (Agent for Agent) — 에이전트를 검증하는 에이전트

**Runtime Policy Gate 기반 검증 가능한 Enterprise AI Agent**
> 원티드 AI Championship 2026 출품작

[🔗 데모 (Streamlit)](https://wanted-ai-hackathon-qbxoq55djqbp7lijjogjfv.streamlit.app) · [GitHub](https://github.com/Hoseong21/wanted-ai-hackathon)

<br>

## 📌 Project Info

| **대회** | 원티드 AI Championship 2026 |
|---|---|
| **일정** | 신청 9/18 · 예선 제출 9/20 · 예선 심사 9/21~10/5 · 본선 발표 10/7 · 데모데이 10/17 |
| **참가자** | 송호성 (개인 프로젝트) |
| **분야** | LangGraph 기반 Enterprise AI Agent + 실행 전 검증 계층(Runtime Policy Gate) |

<br>

## 📖 Overview

LLM 에이전트는 30분 내외의 짧은 과제는 완료율이 약 80%에 달하지만, 여러 시간이 걸리는 과제로 갈수록 완료율이 25% 미만으로 급락한다 (International AI Safety Report 2026). "일을 잘 해내는 것"보다 **"일을 신뢰할 수 있게, 검증 가능하게 해내는 것"**이 기업 환경에서 AI 에이전트 도입의 진짜 병목이라는 문제의식에서 출발한 프로젝트입니다.

사내 구매 승인 업무를 예시 도메인으로 삼아, LangGraph 기반 구매 승인 에이전트를 만들되 **실제 실행(구매 등록) 직전에 런타임 정책 검증 게이트가 개입**해 정책 위반 여부를 재검증하고, 위반 시 차단(BLOCK) 또는 사람 승인 요청(REQUIRE_APPROVAL)으로 라우팅하며, 4종의 평가자(Evaluator)로 에이전트가 실제로 무엇을 했는지 사후 정량 검증합니다.

핵심 차별점은 단순 사후 로깅·모니터링(탐지)이 아니라, **실행 직전 게이트가 실제로 위반을 막는다(방지)**는 것을 14개 시나리오 × 3회 반복 실험으로 정량 실증한 것입니다.

이 카테고리(에이전트 가드레일/평가) 자체는 이미 Galileo AI, Patronus AI, NVIDIA NeMo Guardrails, RAGAS 등 상용 서비스와 학술 연구(arXiv:2603.20953)가 존재합니다. "참신한 문제"가 아니라 **이미 검증된 문제를 직접 구현하고, gate 없는 조건과 있는 조건을 baseline까지 포함해 3단으로 정량 비교해 효과를 실증**했다는 점으로 정직하게 포지셔닝합니다.

<br>

## ⚙️ Architecture

```
User
  │
  ▼
Agent (LangGraph, tool-calling, gpt-5.6-luna)
  │  tools: product_search · policy_search · budget_check ·
  │         purchase_request · purchase_register
  ▼
purchase_register 호출 시 ─────► Runtime Policy Gate (policy_engine.py)
                                   ALLOW / REQUIRE_APPROVAL / BLOCK
                                          │
                          REQUIRE_APPROVAL → interrupt() → Human Approval UI
                                          │            (Streamlit, 승인/거절)
                                          ▼
                                   Command(resume=True/False)로 그래프 재개
                                          │
                                          ▼
                        4 Evaluators (사후 정량 검증, 매 실행마다 기록)
        ┌─────────────┬──────────────────┬───────────────┬──────────────┐
        │ Policy Engine│ Tool Call Check  │ Grounding Check│ Outcome Check│
        │(authoritative│(필요한 도구를    │(답변 속 숫자가 │(최종 결과가  │
        │  정책 판단)  │ 실제로 호출했는가│ 실제 근거에서  │ 시나리오별   │
        │              │/금지 도구 호출은 │ 나왔는가)      │ 기대와       │
        │              │ 없었는가)        │                │ 일치하는가)  │
        └─────────────┴──────────────────┴───────────────┴──────────────┘
```

**판단 없는 도구(judgment-free tools) 설계 원칙**: 도구 자체는 정책·승인 여부를 판단하지 않고 사실 정보만 반환합니다. 판단은 에이전트와 정책 게이트가 전담합니다. `policy_search`는 에이전트가 필요하다고 판단할 때만 호출하는 도구로, 강제 호출이 아니며 호출 여부 자체가 평가 대상입니다.

### Runtime Policy Gate 판정 규칙 (`src/agent/policy_rules.yaml`)

`purchase_register`가 호출되면 LLM이 넘긴 인자를 신뢰하지 않고 `request_id`로 DB를 다시 조회해 재검증합니다 (금액 조작 원천 차단 — `purchase_register`는 구조적으로 `request_id`만 인자로 받음).

1. 금지 카테고리(기프트카드·주류 등) → **BLOCK**
2. 예산 부족 → BLOCK 또는 REQUIRE_APPROVAL
3. 부서장 승인 한도(200만원) 이상 → **REQUIRE_APPROVAL**
4. "리퍼비시"·"중고" 등 제한 키워드 포함 → 금액과 무관하게 **REQUIRE_APPROVAL**
5. 자동승인 한도(50만원) 초과 → **REQUIRE_APPROVAL**
6. 최근 7일 내 동일 카테고리 누적 금액 기준 분할구매 탐지 → **REQUIRE_APPROVAL**
7. 위 전부 통과 → **ALLOW**

### Human-in-the-loop 승인 게이트

`REQUIRE_APPROVAL` 판정 시 LangGraph의 `interrupt({"type": "approval_request", "request_id", "reason", "facts"})`로 그래프 실행이 멈추고, Streamlit UI가 승인/거절 버튼을 렌더링합니다. 사람이 응답하면 동일한 `thread_config`로 `Command(resume=True/False)`를 넘겨 그래프를 재개합니다. 한 턴 안에서 여러 번의 interrupt가 연속될 수 있어 `st.session_state`로 진행 상태를 관리합니다.

<br>

## 🔍 4-Evaluator 역할 분리

| Evaluator | 역할 |
|---|---|
| **Policy Engine** | 정책 판단의 authoritative 진실 소스 (YAML 규칙 기반) |
| **Tool Call Check** | 필요한 도구를 실제로 조회했는가 / 금지된 도구를 호출했는가 |
| **Grounding Check** | 답변에 언급된 숫자가 실제 관측된 근거에서 나왔는가 (3단계 근거 모델) |
| **Outcome Check** | 최종 결과가 시나리오별 기대 결과(등록/승인대기/차단)와 일치하는가 |

**Grounding Check — 3단계 근거 모델**
- Level 1 (직접 근거): 숫자가 도구 결과 JSON에 그대로 등장
- Level 2 (파생 근거, 화이트리스트 2종만 허용): `budget_shortfall = price - remaining_budget`, `welfare_budget_limit = team_budget × welfare_ratio(%)` — 일반 산술 유추는 허용하지 않고 이 두 케이스만 명시적으로 허용
- Level 3: 근거 없음 → 실패

"라벨: 숫자원" 형태의 명시적 클레임만 의미 검증하고, 자유 서술문·쉼표 나열은 존재 여부만 확인합니다 — **판단 유보가 오탐(false positive)보다 낫다**는 원칙으로 스코프를 의도적으로 좁혔습니다.

<br>

## 📊 실험 결과

14개 시나리오(S01~S14, 정상 케이스 + 경계값 + 금지품목 + 분할구매 + 정보누락 + 허위승인주장 등 의도적 실패 유도 포함) × 3회 반복 = 총 98회 실행(baseline 14회 + no-gate 42회 + gated 42회).

| 지표 | Gate 없음 | Gate 있음 |
|---|---|---|
| 정책 위반 발생 | **27 / 42** | **0 / 42** |
| 등록 시도(register_attempted) | 36 / 42 | 36 / 42 |
| Outcome 기대 결과 일치 | 15 / 42 | 42 / 42 |
| 4종 검증 전체 통과 (tool call + grounding + outcome) | — | 39 / 42 |

**핵심 메시지**: 동일한 14개 시나리오에서 게이트가 없으면 42회 실행 중 27회 실제 정책 위반이 발생했으나, 게이트를 켜자 위반이 0건으로 사라졌습니다.

> ⚠️ **스코프 명시**: 위 수치는 "이 14개 synthetic 시나리오, 3회 반복 실험(rule-based evaluator 기준)"에서의 결과이며, "AfA가 정책 위반을 100% 방지한다"는 식의 일반화된 주장은 하지 않습니다.

### 알려진 한계

1. **게이트는 최종 등록(`purchase_register`) 시점에만 개입합니다.** 그 이전 단계의 잘못된 도구 호출은 막지 못하고 사후에만 탐지됩니다. 실제로 `S04_금지품목`(주류 등 금지 품목 구매) 시나리오에서 이 한계가 관측되었습니다 — 게이트가 최종 등록은 3/3 모두 정상적으로 차단했지만(정책 위반 0건), 에이전트가 금지 품목인지 확인하기 전에 `purchase_request`(구매 요청 접수)를 먼저 호출해버려 `tool_call_check`가 3/3 실패했습니다. "결과적으로 사고는 막았지만, 과정 자체는 이상적이지 않았다"는 케이스입니다.
2. Grounding Check는 쉼표 나열·자유 서술문에는 의미적 라벨 검사를 적용하지 않습니다(존재 여부만 확인).
3. 엔티티 단위 출처 추적이 없습니다 (예: 서로 다른 팀의 잔여예산 숫자가 하나의 근거 집합으로 합쳐짐).
4. `welfare_budget_limit` 파생값의 퍼센트 추출은 키워드 앵커링 방식이라, 같은 문맥에 퍼센트가 여러 개 등장하면 느슨해질 수 있습니다 (현재 정책 문서 기준으로는 해당 없음).
5. 시나리오는 합성 데이터이며 14개로 카테고리 커버리지 중심입니다. 실제 사내 데이터/더 넓은 카테고리에 대한 일반화는 검증되지 않았습니다.

<br>

## 🛠 Tech Stack

**Agent Orchestration** : LangGraph 1.2.11 · LangChain 1.4.0 (직접 그래프 구현, `create_react_agent` 미사용 — 게이트 개입 지점의 투명성을 위해)

**LLM** : OpenAI (`reasoning_effort="none"`, `temperature=0` — 재현성)

**RAG** : ChromaDB · KURE-v1(1단계 검색) → KURE-v2(2단계 재순위, retrieve-then-rerank) · sentence-transformers

**DB** : SQLite (팀별 예산)

**Frontend** : Streamlit (Human Approval UI + 실행 트레이스 시각화)

<br>

## 🗂 프로젝트 구조

```
data/            가짜 사내 구매규정 문서(6종), mock 제품 카탈로그, 예산 DB 초기화
src/rag/         문서 적재(ingest) 및 검색(retriever, KURE 임베딩/재순위)
src/tools/       에이전트가 호출하는 5종 도구
src/agent/       LangGraph 그래프, 정책 게이트(policy_engine.py), 정책 규칙(policy_rules.yaml)
src/evaluators/  Tool Call / Grounding / Outcome 3종 검증기
src/logging/     실행 액션 로깅
app/             Streamlit 기반 Human Approval UI
experiments/     14개 시나리오 + baseline/no-gate/gated 3조건 비교 실험 하네스
```

<br>

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env에 OPENAI_API_KEY 입력

# 3. 예산 DB 초기화
python3 -m data.init_budget_db

# 4. 정책 문서 벡터스토어 적재
python3 -m src.rag.ingest

# 5-a. 로컬에서 Streamlit 앱 실행
streamlit run app/streamlit_app.py

# 5-b. 또는 baseline/no-gate/gated 비교 실험 재현
REPEATS=3 python3 experiments/run_comparison.py
# 결과는 experiments/results_repeated.json 에 저장됩니다
```

배포된 Streamlit Cloud 앱은 로그인·API 키 입력 없이 바로 체험할 수 있도록 구성했습니다 (OpenAI API 키는 Streamlit Cloud Secrets에 서버 측으로 보관).

🔗 **데모**: https://wanted-ai-hackathon-qbxoq55djqbp7lijjogjfv.streamlit.app