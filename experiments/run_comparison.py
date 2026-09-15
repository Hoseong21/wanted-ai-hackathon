"""baseline / agent(게이트 없음) / agent+verification(게이트 있음) 3조건 비교 실행 스크립트.

시나리오마다 (b) 게이트 없는 에이전트와 (c) 게이트 있는 에이전트를 각각 독립된 임시 DB에서
실행해서 "게이트가 없었으면 위반이 실제로 일어났을지"를 항상 직접 비교할 수 있게 한다.
(모델이 우연히 조심스럽게 행동해도 게이트의 효과를 놓치지 않기 위함.)

주의: 시나리오당 (b)+(c) 두 번의 에이전트 실행이 필요해서 API 호출이 이전보다 늘어난다.
"""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src import config
from src.agent.baseline import run_baseline
from src.agent.graph import build_graph
from src.evaluators.grounding_check import extract_currency_numbers, get_final_answer
from src.evaluators.grounding_check import grounding_check as grounding_check_fn
from src.evaluators.outcome_check import outcome_check as outcome_check_fn
from src.evaluators.tool_call_check import tool_call_check as tool_call_check_fn

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from init_budget_db import init_db  # noqa: E402  # type: ignore

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


@contextmanager
def temp_budget_db():
    tmp_path = Path(tempfile.mkstemp(suffix=".db")[1])
    try:
        init_db(db_path=tmp_path, reset=True)
        config.BUDGET_DB_PATH = tmp_path
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def _check_baseline_grounding(baseline_result: dict) -> dict:
    answer = baseline_result["answer"]
    pool_text = "\n".join(h["text"] for h in baseline_result["retrieved_context"])
    claimed = extract_currency_numbers(answer)
    pool_numbers = extract_currency_numbers(pool_text)
    ungrounded = sorted(n for n in claimed if n not in pool_numbers)
    return {"passed": not ungrounded, "claimed_numbers": sorted(claimed), "ungrounded_numbers": ungrounded}


def run_scenario(scenario: dict) -> dict:
    query = scenario["user_query"]

    baseline_result = run_baseline(query)
    baseline_grounding = _check_baseline_grounding(baseline_result)

    # (b) Agent 단독 (게이트 없음)
    with temp_budget_db():
        agent_app_nogate = build_graph(gated=False)
        thread_nogate = {"configurable": {"thread_id": f"{scenario['scenario_id']}-nogate"}}
        result_nogate = agent_app_nogate.invoke({"messages": [HumanMessage(content=query)]}, config=thread_nogate)
    messages_nogate = result_nogate["messages"]
    outcome_nogate = outcome_check_fn(messages_nogate, scenario["expected_outcome"])

    # (c) Agent + Verification (게이트 있음). REQUIRE_APPROVAL은 스크립트 승인자가 거절 처리.
    with temp_budget_db():
        agent_app_gated = build_graph(gated=True)
        thread_gated = {"configurable": {"thread_id": f"{scenario['scenario_id']}-gated"}}
        result_gated = agent_app_gated.invoke({"messages": [HumanMessage(content=query)]}, config=thread_gated)
        while "__interrupt__" in result_gated:
            result_gated = agent_app_gated.invoke(Command(resume=False), config=thread_gated)
    messages_gated = result_gated["messages"]

    tcc = tool_call_check_fn(messages_gated, scenario["expected_tool_calls"])
    gc = grounding_check_fn(messages_gated)
    oc = outcome_check_fn(messages_gated, scenario["expected_outcome"])

    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "user_query": query,
        "baseline": {"answer": baseline_result["answer"], "grounding": baseline_grounding},
        "agent_no_gate": {
            "final_answer": get_final_answer(messages_nogate),
            "outcome_check": outcome_nogate,
        },
        "agent_with_gate": {
            "final_answer": get_final_answer(messages_gated),
            "called_tools": tcc["called_tools"],
            "executed_tools": tcc["executed_tools"],
        },
        "verification": {
            "tool_call_check": tcc,
            "grounding_check": gc,
            "outcome_check": oc,
            "all_passed": tcc["passed"] and gc["passed"] and oc["passed"],
        },
        "gate_effect": {
            "violation_without_gate": not outcome_nogate["passed"],
            "violation_with_gate": not oc["passed"],
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
        "agent_outcome_check_pass_rate": rate(lambda r: r["verification"]["outcome_check"]["passed"]),
        "agent_overall_pass_rate": rate(lambda r: r["verification"]["all_passed"]),
        "violations_without_gate": sum(1 for r in results if r["gate_effect"]["violation_without_gate"]),
        "violations_with_gate": sum(1 for r in results if r["gate_effect"]["violation_with_gate"]),
    }


def main() -> None:
    scenarios = load_scenarios()
    results = []
    for scenario in scenarios:
        print(f"실행 중: {scenario['scenario_id']}...")
        results.append(run_scenario(scenario))

    summary = summarize(results)
    RESULTS_PATH.write_text(
        json.dumps({"results": results, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== 요약 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n상세 결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    main()