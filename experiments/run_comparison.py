"""baseline / agent / agent+verification 3조건 비교 실행 스크립트.

experiments/scenarios.json의 시나리오들을 세 조건으로 실행하고, 조건별 결과와
(agent에 대한) 평가기 3종 결과를 모아 experiments/results.json으로 저장한다.
집계 통계도 함께 콘솔에 출력한다.

주의: 실행할 때마다 실제 OpenAI API를 호출한다 (시나리오당 baseline 1회 + agent
3~5회 정도 호출 → 전체 6개 시나리오 기준 대략 25~30회 호출).

용어 정정: 기획서상 "정책위반 방지율"이라고 돼있지만, 현재 아키텍처는 에이전트가
purchase_register를 실제로 실행한 뒤 평가기가 사후 검증하는 구조라 (Human Approval
게이트 미구현) 정확히는 "탐지율"이다. 이 스크립트의 지표도 탐지율 기준으로 계산한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent.baseline import run_baseline
from src.agent.graph import build_graph
from src.evaluators.grounding_check import extract_currency_numbers, get_final_answer
from src.evaluators.grounding_check import grounding_check as grounding_check_fn
from src.evaluators.policy_check import policy_check as policy_check_fn
from src.evaluators.tool_call_check import tool_call_check as tool_call_check_fn

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def _check_baseline_grounding(baseline_result: dict) -> dict:
    """baseline은 LangGraph 메시지 트레이스가 없어서 grounding_check를 그대로 못 쓰고,
    같은 원리(답변의 금액이 검색된 문서에 실제로 있는지)만 재사용해서 직접 체크한다."""
    answer = baseline_result["answer"]
    pool_text = "\n".join(h["text"] for h in baseline_result["retrieved_context"])

    claimed = extract_currency_numbers(answer)
    pool_numbers = extract_currency_numbers(pool_text)
    ungrounded = sorted(n for n in claimed if n not in pool_numbers)

    return {
        "passed": not ungrounded,
        "claimed_numbers": sorted(claimed),
        "ungrounded_numbers": ungrounded,
    }


def run_scenario(scenario: dict, agent_app) -> dict:
    query = scenario["user_query"]

    # (a) RAG baseline
    baseline_result = run_baseline(query)
    baseline_grounding = _check_baseline_grounding(baseline_result)

    # (b) Agent - 검증 없이 바로 실행
    agent_result = agent_app.invoke({"messages": [HumanMessage(content=query)]})
    messages = agent_result["messages"]
    final_answer = get_final_answer(messages)

    # (c) Agent + Verification - (b)의 트레이스에 평가기 3종 적용
    tcc = tool_call_check_fn(messages, scenario["expected_tool_calls"])
    gc = grounding_check_fn(messages)
    pc = policy_check_fn(messages, scenario["expected_outcome"])

    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "user_query": query,
        "baseline": {
            "answer": baseline_result["answer"],
            "grounding": baseline_grounding,
        },
        "agent": {
            "final_answer": final_answer,
            "called_tools": tcc["called_tools"],
        },
        "verification": {
            "tool_call_check": tcc,
            "grounding_check": gc,
            "policy_check": pc,
            "all_passed": tcc["passed"] and gc["passed"] and pc["passed"],
        },
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)

    def rate(pred) -> float:
        return sum(1 for r in results if pred(r)) / n

    return {
        "total_scenarios": n,
        "baseline_grounding_pass_rate": rate(lambda r: r["baseline"]["grounding"]["passed"]),
        "agent_tool_call_check_pass_rate": rate(lambda r: r["verification"]["tool_call_check"]["passed"]),
        "agent_grounding_check_pass_rate": rate(lambda r: r["verification"]["grounding_check"]["passed"]),
        "agent_policy_check_pass_rate": rate(lambda r: r["verification"]["policy_check"]["passed"]),
        "agent_overall_pass_rate": rate(lambda r: r["verification"]["all_passed"]),
    }


def main() -> None:
    scenarios = load_scenarios()
    agent_app = build_graph()

    results = []
    for scenario in scenarios:
        print(f"실행 중: {scenario['scenario_id']}...")
        results.append(run_scenario(scenario, agent_app))

    summary = summarize(results)

    RESULTS_PATH.write_text(
        json.dumps({"results": results, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 요약 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n상세 결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    main()