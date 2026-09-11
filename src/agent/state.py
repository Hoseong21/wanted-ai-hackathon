"""에이전트의 LangGraph 상태(state) 정의.

메시지 목록 하나만 관리한다. 각 메시지(HumanMessage/AIMessage/ToolMessage)에 이미
"에이전트가 어떤 툴을 어떤 인자로 호출했고 결과가 뭐였는지"가 다 기록되기 때문에,
지금 단계에서는 별도 필드를 추가할 필요가 없다. 나중에 evaluators가 이 메시지 목록을
그대로 읽어서 grounding_check/tool_call_check를 수행한다.
"""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """LangGraph 표준 MessagesState 그대로 사용. 확장 필요해지면 여기에 필드 추가."""