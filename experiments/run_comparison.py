"""baseline / agent(게이트 없음) / agent+verification(게이트 있음) 3조건 비교 실행 스크립트.

시나리오마다 (b) 게이트 없는 에이전트와 (c) 게이트 있는 에이전트를 각각 독립된 임시 DB에서
실행해서 "게이트가 없었으면 위반이 실제로 일어났을지"를 항상 직접 비교할 수 있게 한다.
(모델이 우연히 조심스럽게 행동해도 게이트의 효과를 놓치지 않기 위함.)

주의: 시나리오당 (b)+(c) 두 번의 에이전트 실행이 필요해서 API 호출이 이전보다 늘어난다.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import os
import json
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src import config
from src.agent.baseline import run_baseline
from src.agent.graph import build_graph
from src.db.init_budget_db import create_temp_budget_db
from src.evaluators.grounding_check import extract_currency_numbers, get_final_answer
from src.evaluators.grounding_check import grounding_check as grounding_check_fn
from src.evaluators.outcome_check import outcome_check as outcome_check_fn
from src.evaluators.tool_call_check import tool_call_check as tool_call_check_fn
from src.evaluators.tool_call_check import extract_called_tools

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


@contextmanager
def temp_budget_db():
    """시나리오 실행마다 독립된 임시 budget DB를 만든다.

    전역 config.BUDGET_DB_PATH는 건드리지 않는다. 대신 yield된 경로를
    호출부가 apply_initial_state()와 thread_config의 configurable.db_path에
    명시적으로 넘겨서, RunnableConfig를 통해 툴/그래프 실행에 주입한다.
    """
    tmp_path = create_temp_budget_db()
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def apply_initial_state(initial_state: dict | None, products_by_id: dict, db_path: Path) -> None:
    """시나리오의 initial_state를 지정된 db_path(임시 DB)에 적용한다.
    temp_budget_db()로 DB가 만들어진 직후, 에이전트 실행 전에 호출해야 한다.
    """
    if not initial_state:
        return
    conn = sqlite3.connect(db_path)
    try:
        for ov in initial_state.get("budget_overrides", []):
            conn.execute(
                "UPDATE budget SET spent_amount = ? WHERE team_id = "
                "(SELECT team_id FROM teams WHERE team_name = ?)",
                (ov["spent_amount"], ov["team_name"]),
            )
        for seed in initial_state.get("seed_purchase_requests", []):
            product = products_by_id[seed["product_id"]]
            quantity = seed.get("quantity", 1)
            unit_price = product["price"]
            total_price = unit_price * quantity
            ts = (datetime.now(timezone.utc) - timedelta(days=seed.get("days_ago", 3))).isoformat()
            registered_at = ts if seed.get("registered", True) else None
            conn.execute(
                "INSERT INTO purchase_requests "
                "(team_name, product_id, quantity, requester, unit_price, total_price, created_at, registered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seed["team_name"], seed["product_id"], quantity, seed.get("requester", "기존직원"),
                 unit_price, total_price, ts, registered_at),
            )
            if registered_at:
                conn.execute(
                    "UPDATE budget SET spent_amount = spent_amount + ? WHERE team_id = "
                    "(SELECT team_id FROM teams WHERE team_name = ?)",
                    (total_price, seed["team_name"]),
                )
        conn.commit()
    finally:
        conn.close()


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def _check_baseline_grounding(baseline_result: dict) -> dict:
    answer = baseline_result["answer"]
    pool_text = "\n".join(h["text"] for h in baseline_result["retrieved_context"])
    claimed = extract_currency_numbers(answer)
    pool_numbers = extract_currency_numbers(pool_text)
    ungrounded = sorted(n for n in claimed if n not in pool_numbers)
    return {"passed": not ungrounded, "claimed_numbers": sorted(claimed), "ungrounded_numbers": ungrounded}


def run_scenario(scenario: dict, repeat_idx: int = 0, run_baseline_flag: bool = True) -> dict:
    query = scenario["user_query"]
    products_by_id = {p["id"]: p for p in json.loads(config.PRODUCTS_PATH.read_text(encoding="utf-8"))}

    baseline_section = None
    if run_baseline_flag:
        baseline_result = run_baseline(query)
        baseline_grounding = _check_baseline_grounding(baseline_result)
        baseline_section = {"answer": baseline_result["answer"], "grounding": baseline_grounding}

    # (b) Agent 단독 (게이트 없음)
    with temp_budget_db() as db_path_nogate:
        apply_initial_state(scenario.get("initial_state"), products_by_id, db_path_nogate)
        agent_app_nogate = build_graph(gated=False)
        thread_nogate = {
            "configurable": {
                "thread_id": f"{scenario['scenario_id']}-nogate-r{repeat_idx}",
                "scenario_id": scenario["scenario_id"],
                "db_path": db_path_nogate,
            }
        }
        result_nogate = agent_app_nogate.invoke({"messages": [HumanMessage(content=query)]}, config=thread_nogate)
    messages_nogate = result_nogate["messages"]
    outcome_nogate = outcome_check_fn(messages_nogate, scenario["expected_outcome"])
    called_nogate = extract_called_tools(messages_nogate)

    # (c) Agent + Verification (게이트 있음)
    with temp_budget_db() as db_path_gated:
        apply_initial_state(scenario.get("initial_state"), products_by_id, db_path_gated)
        agent_app_gated = build_graph(gated=True)
        thread_gated = {
            "configurable": {
                "thread_id": f"{scenario['scenario_id']}-gated-r{repeat_idx}",
                "scenario_id": scenario["scenario_id"],
                "db_path": db_path_gated,
            }
        }
        result_gated = agent_app_gated.invoke({"messages": [HumanMessage(content=query)]}, config=thread_gated)
        while "__interrupt__" in result_gated:
            result_gated = agent_app_gated.invoke(Command(resume=False), config=thread_gated)
    messages_gated = result_gated["messages"]

    tcc = tool_call_check_fn(messages_gated, scenario["expected_tool_calls"])
    gc = grounding_check_fn(messages_gated)
    oc = outcome_check_fn(messages_gated, scenario["expected_outcome"])

    result = {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "user_query": query,
        "agent_no_gate": {
            "final_answer": get_final_answer(messages_nogate),
            "called_tools": called_nogate,
            "register_attempted": "purchase_register" in called_nogate,
            "outcome_check": outcome_nogate,
        },
        "agent_with_gate": {
            "final_answer": get_final_answer(messages_gated),
            "called_tools": tcc["called_tools"],
            "executed_tools": tcc["executed_tools"],
            "register_attempted": "purchase_register" in tcc["called_tools"],
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
    if baseline_section is not None:
        result["baseline"] = baseline_section
    return result


REPEATS = int(os.environ.get("REPEATS", "3"))


def run_scenario_repeated(scenario: dict, repeats: int) -> dict:
    runs = [run_scenario(scenario, repeat_idx=i, run_baseline_flag=(i == 0)) for i in range(repeats)]

    def count(pred) -> int:
        return sum(1 for r in runs if pred(r))

    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "user_query": scenario["user_query"],
        "repeats": repeats,
        "baseline": runs[0].get("baseline"),
        "aggregate": {
            "violations_without_gate": count(lambda r: r["gate_effect"]["violation_without_gate"]),
            "violations_with_gate": count(lambda r: r["gate_effect"]["violation_with_gate"]),
            "register_attempted_without_gate": count(lambda r: r["agent_no_gate"]["register_attempted"]),
            "register_attempted_with_gate": count(lambda r: r["agent_with_gate"]["register_attempted"]),
            "outcome_success_without_gate": count(lambda r: r["agent_no_gate"]["outcome_check"]["passed"]),
            "outcome_success_with_gate": count(lambda r: r["verification"]["outcome_check"]["passed"]),
            "full_verification_pass_with_gate": count(lambda r: r["verification"]["all_passed"]),
        },
        "runs": runs,
    }


def summarize(scenario_aggregates: list[dict]) -> dict:
    n = len(scenario_aggregates)
    total_runs = sum(1 + 2 * s["repeats"] for s in scenario_aggregates)  # baseline 1 + no-gate N + gated N

    def total(key: str) -> int:
        return sum(s["aggregate"][key] for s in scenario_aggregates)

    return {
        "total_scenarios": n,
        "repeats_per_scenario": REPEATS,
        "total_condition_executions": total_runs,
        "violations_without_gate_total": total("violations_without_gate"),
        "violations_with_gate_total": total("violations_with_gate"),
        "register_attempted_without_gate_total": total("register_attempted_without_gate"),
        "register_attempted_with_gate_total": total("register_attempted_with_gate"),
        "outcome_success_without_gate_total": total("outcome_success_without_gate"),
        "outcome_success_with_gate_total": total("outcome_success_with_gate"),
        "full_verification_pass_with_gate_total": total("full_verification_pass_with_gate"),
    }


def main() -> None:
    scenarios = load_scenarios()
    scenario_aggregates = []
    for scenario in scenarios:
        print(f"실행 중: {scenario['scenario_id']} ({REPEATS}회 반복)...")
        scenario_aggregates.append(run_scenario_repeated(scenario, REPEATS))

    summary = summarize(scenario_aggregates)
    REPEATED_RESULTS_PATH = Path(__file__).parent / "results_repeated.json"
    REPEATED_RESULTS_PATH.write_text(
        json.dumps({"scenarios": scenario_aggregates, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n===== 요약 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n상세 결과 저장: {REPEATED_RESULTS_PATH}")


if __name__ == "__main__":
    main()