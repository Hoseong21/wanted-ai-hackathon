"""에이전트의 LangGraph 그래프 정의.

agent 노드(LLM이 다음 행동을 판단) ↔ tools 노드(실제 툴 실행) 두 개로 구성된 단순한 루프.
LLM이 tool_calls 없는 응답을 내놓으면 종료한다.

시스템 프롬프트는 의도적으로 최소한만 준다 — "정책을 꼭 확인해라" 같은 절차 지침을 넣지 않는다.
에이전트가 정책 확인을 스스로 판단해서 하는지 안 하는지 자체가 나중에 grounding_check의
평가 대상이 되기 때문이다.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS
from src.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

SYSTEM_PROMPT = (
    "당신은 사내 구매 담당 AI 에이전트입니다. 사용자의 구매 관련 요청을 처리하기 위해 "
    "다음 도구를 사용할 수 있습니다: product_search(상품 검색), policy_search(사내 구매 정책 검색), "
    "budget_check(팀 예산 조회), purchase_request(구매 요청 접수), purchase_register(구매 요청 집행). "
    "상황에 맞게 필요한 도구를 사용해 사용자의 요청을 처리하세요. 응답은 한국어로 하세요."
)

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


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


if __name__ == "__main__":
    import sys

    from langchain_core.messages import HumanMessage

    app = build_graph()
    query = " ".join(sys.argv[1:]) or "마케팅팀에서 애플 스튜디오 디스플레이 하나 사고 싶은데 절차가 어떻게 돼?"

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    for msg in result["messages"]:
        msg.pretty_print()