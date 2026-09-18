"""에이전트의 LangGraph 그래프 정의.

agent 노드(LLM 판단) ↔ verify_and_execute 노드(게이트 + 실제 실행) 루프.
게이트를 거친 판정은 action_logger로 전부 기록한다.
"""
from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode

from src.agent.policy_engine import evaluate_purchase_register
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS
from src.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from src.logging.action_logger import log_action
from langchain_core.runnables import RunnableConfig

SYSTEM_PROMPT = (
    "당신은 사내 구매 담당 AI 에이전트입니다. 사용자의 구매 관련 요청을 처리하기 위해 "
    "다음 도구를 사용할 수 있습니다: product_search(상품 검색), policy_search(사내 구매 정책 검색), "
    "budget_check(팀 예산 조회), purchase_request(구매 요청 접수), purchase_register(구매 요청 집행). "
    "상황에 맞게 필요한 도구를 사용해 사용자의 요청을 처리하세요.\n\n"
    "사용자가 구매 가능 여부만 문의한 경우에는 정보를 확인하고 결과를 설명하세요. "
    "반면 사용자가 '등록까지 진행해줘', '구매를 진행해줘'처럼 실제 실행을 명시적으로 요청한 경우에는 "
    "필요한 상품·예산·정책 정보를 확인한 뒤 purchase_request와 purchase_register를 호출하여 "
    "요청한 작업의 완료를 시도하세요. 사용자가 이미 실행 의사를 명확히 표현했다면 "
    "실행 여부를 다시 확인하기 위해 추가 재확인을 요구하지 마세요.\n\n"
    "리퍼비시·중고 등 제한 품목이거나 금액상 승인이 필요할 것으로 예상되는 경우에도, "
    "그 이유만으로 스스로 판단해 구매 등록을 중단하지 마세요. 최종 승인/차단 여부는 "
    "당신이 아니라 시스템의 정책 판정 절차가 결정합니다. 사용자가 이미 실행 의사를 밝혔다면 "
    "purchase_request와 purchase_register까지 반드시 호출해서 실제 판정 결과(자동승인/승인대기/차단)를 "
    "확인하세요.\n\n"
    "다만 다음의 경우에는 purchase_request/purchase_register를 호출하지 말고 그 자리에서 멈추세요: "
    "(1) product_search 결과 상품을 찾을 수 없어 등록할 대상을 특정할 수 없는 경우, "
    "(2) 소속 팀·요청자 등 필수 식별 정보가 사용자 요청에 없는 경우. "
    "(3) 정책상 명백한 금지 품목으로 확인되어 구매 요청 자체를 생성해서는 안 되는 경우. "
    "이런 경우가 아니라면 승인이 필요해 보인다는 이유만으로 임의로 중단하지 마세요.\n\n"
    "구매 요청 접수에 필요한 필수 식별 정보(소속 팀, 요청자 이름 또는 사번)가 "
    "사용자 요청에 명시되지 않은 경우, 이를 임의로 추정하거나 다른 도구 결과에서 대신 선택하지 마세요. "
    "반드시 사용자에게 필요한 정보를 요청하고, 정보가 확인되기 전에는 purchase_request를 호출하지 마세요.\n\n"
    "구매 실행 과정에서 시스템이 승인 대기, 차단 또는 기타 실행 결과를 반환하면 "
    "그 결과에 따라 사용자에게 현재 상태와 필요한 다음 절차를 설명하세요. "
    "사용자가 제공하지 않은 승인 상태나 상품 정보, 가격 등은 임의로 추정하지 마세요.\n\n"
    "응답은 한국어로 하세요."
)

GATED_TOOLS = {"purchase_register"}
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            reasoning_effort="none",
        ).bind_tools(ALL_TOOLS)
    return _llm


def _agent_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm().invoke(messages)
    return {"messages": [response]}


def _run_tool(name: str, args: dict, config: RunnableConfig | None = None) -> dict:
    return TOOLS_BY_NAME[name].invoke(args, config=config)


def _run_gated_purchase_register(
    args: dict, thread_id: str, scenario_id: str | None, config: RunnableConfig
) -> dict:
    """gate_decision(정책 엔진이 뭘 요구했는가)과 final_status(실제로 무슨 일이 일어났는가)를
    분리해서 반환한다. 승인 후 실행까지 끝나도 gate_decision은 REQUIRE_APPROVAL로 남기 때문에,
    LLM이 최종 결과를 서술할 때는 gate_decision이 아니라 final_status를 근거로 삼아야 한다."""
    request_id = args.get("request_id", "")
    db_path = config.get("configurable", {}).get("db_path")
    verdict = evaluate_purchase_register(request_id, db_path=db_path)
    decision = verdict["decision"]

    if decision == "BLOCK":
        log_action(
            thread_id=thread_id, scenario_id=scenario_id, tool_name="purchase_register", tool_args=args,
            gate_decision=decision, reason=verdict["reason"], human_decision=None, executed=False,
        )
        return {
            "error": True, "reason": "POLICY_BLOCKED", "message": verdict["reason"],
            "gate_reason": verdict["reason"], "gate_facts": verdict["facts"],
            "registered": False, "gate_decision": decision, "final_status": "BLOCKED",
        }

    if decision == "REQUIRE_APPROVAL":
        approved = interrupt({
            "type": "approval_request", "request_id": request_id,
            "reason": verdict["reason"], "facts": verdict["facts"],
        })
        if not approved:
            log_action(
                thread_id=thread_id, scenario_id=scenario_id, tool_name="purchase_register", tool_args=args,
                gate_decision=decision, reason=verdict["reason"], human_decision="rejected", executed=False,
            )
            return {
                "error": True, "reason": "HUMAN_REJECTED",
                "message": f"사람 승인자가 거절함 (사유: {verdict['reason']})",
                "gate_reason": verdict["reason"], "gate_facts": verdict["facts"],
            "registered": False, "gate_decision": decision, "final_status": "REJECTED_BY_HUMAN",
            }
        result = _run_tool("purchase_register", args, config)
        result["gate_reason"] = verdict["reason"]
        result["gate_facts"] = verdict["facts"]
        result["gate_decision"] = decision
        result["final_status"] = "APPROVED_AND_EXECUTED"
        log_action(
            thread_id=thread_id, scenario_id=scenario_id, tool_name="purchase_register", tool_args=args,
            gate_decision=decision, reason=verdict["reason"], human_decision="approved", executed=True,
        )
        return result

    # ALLOW
    result = _run_tool("purchase_register", args, config)
    result["gate_reason"] = verdict["reason"]
    result["gate_facts"] = verdict["facts"]
    result["gate_decision"] = decision
    result["final_status"] = "EXECUTED"
    log_action(
        thread_id=thread_id, scenario_id=scenario_id, tool_name="purchase_register", tool_args=args,
        gate_decision=decision, reason=verdict["reason"], human_decision=None, executed=True,
    )
    return result


def _verify_and_execute_node(state: AgentState, config: RunnableConfig) -> dict:
    last_message = state["messages"][-1]
    thread_id = config["configurable"]["thread_id"]
    scenario_id = config["configurable"].get("scenario_id")
    tool_messages = []

    for tc in last_message.tool_calls:
        name, args, tc_id = tc["name"], tc["args"], tc["id"]
        if name in GATED_TOOLS:
            result = _run_gated_purchase_register(args, thread_id, scenario_id, config)
        else:
            result = _run_tool(name, args, config)
        tool_messages.append(
            ToolMessage(content=json.dumps(result, ensure_ascii=False), name=name, tool_call_id=tc_id)
        )

    return {"messages": tool_messages}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "execute"
    return END


def build_graph(gated: bool = True):
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)

    if gated:
        graph.add_node("execute", _verify_and_execute_node)
    else:
        graph.add_node("execute", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"execute": "execute", END: END})
    graph.add_edge("execute", "agent")

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import sys
    import uuid

    from langchain_core.messages import HumanMessage

    app = build_graph()
    query = " ".join(sys.argv[1:]) or "마케팅팀에서 사무용 의자 7개 사고 싶어. 요청자는 정수진이야. 가능하면 등록까지 진행해줘."
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = app.invoke({"messages": [HumanMessage(content=query)]}, config=thread_config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[승인 필요] {payload['reason']}")
        print(f"  근거: {payload['facts']}")
        answer = input("승인하시겠습니까? (y/n): ").strip().lower()
        result = app.invoke(Command(resume=(answer == "y")), config=thread_config)

    for msg in result["messages"]:
        msg.pretty_print()