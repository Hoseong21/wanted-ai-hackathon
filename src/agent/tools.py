"""5개 mock 툴을 LangChain/LangGraph가 이해하는 tool 객체로 감싸는 어댑터.

src/tools/*.py의 순수 함수들은 그대로 두고, 여기서 LangChain 프레임워크에 필요한
스키마(Literal enum, description)를 입혀서 에이전트가 바인딩할 수 있는 형태로 만든다.
"""

from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.budget_check import budget_check as _budget_check
from src.tools.policy_search import policy_search as _policy_search
from src.tools.product_search import product_search as _product_search
from src.tools.purchase_register import purchase_register as _purchase_register
from src.tools.purchase_request import purchase_request as _purchase_request

TeamName = Literal["AI개발팀", "마케팅팀", "인사팀", "재무팀"]
Category = Literal["노트북", "모니터", "주변기기", "소프트웨어", "사무용품", "기타", "금지품목", "복지"]


def _db_path_from(config: RunnableConfig):
    """RunnableConfig의 configurable.db_path를 꺼낸다.

    없으면 None을 반환해서 하위 함수가 자체적으로 config.BUDGET_DB_PATH로 폴백하게 한다.
    """
    if not config:
        return None
    return config.get("configurable", {}).get("db_path")


class ProductSearchInput(BaseModel):
    query: str = Field(description="검색할 상품명 또는 키워드 (부분 일치, 대소문자 구분 안 함)")
    category: Optional[Category] = Field(
        default=None,
        description=(
            "카테고리 필터. 상품의 정확한 카테고리를 이미 확실히 알고 있는 경우에만 지정하세요. "
            "조금이라도 불확실하면 반드시 생략하고 query만으로 검색한 뒤, "
            "결과에 포함된 category 값으로 실제 분류를 확인하세요."
        ),
    )
    max_price: Optional[int] = Field(default=None, description="최대 가격(원) 필터. 지정 안 하면 제한 없음")


@tool("product_search", args_schema=ProductSearchInput)
def product_search_tool(query: str, category: str | None = None, max_price: int | None = None) -> dict:
    """상품명/키워드로 상품 카탈로그를 검색한다. 상품의 가격, 카테고리, 재고 여부를 알고 싶을 때 사용한다."""
    result = _product_search(query, category=category, max_price=max_price)

    # LLM이 category를 잘못 추측해 0건이 나오는 경우를 방어: category만 제거하고 재검색한다.
    # max_price는 사용자가 실제로 요구한 제약일 수 있으므로 자동으로 풀지 않는다.
    if result.get("count") == 0 and category is not None:
        relaxed = _product_search(query, category=None, max_price=max_price)
        if relaxed.get("count", 0) > 0:
            relaxed["fallback_applied"] = True
            relaxed["relaxed_filters"] = ["category"]
            relaxed["note"] = "category 필터를 제거하고 동일한 가격 조건으로 재검색했습니다."
            return relaxed

    return result


class PolicySearchInput(BaseModel):
    query: str = Field(description="검색할 정책 관련 질문/키워드 (예: '50만원 이상 구매 승인')")
    top_k: int = Field(default=3, description="반환할 최대 결과 수")


@tool("policy_search", args_schema=PolicySearchInput)
def policy_search_tool(query: str, top_k: int = 3) -> dict:
    """사내 구매 정책 문서에서 관련 규정을 검색한다. 구매 승인 기준, 금지 품목, 카테고리별 한도 등을 확인할 때 사용한다."""
    return _policy_search(query, top_k=top_k)


class BudgetCheckInput(BaseModel):
    team_name: TeamName = Field(description="예산을 조회할 팀명")


@tool("budget_check", args_schema=BudgetCheckInput)
def budget_check_tool(team_name: str, config: RunnableConfig) -> dict:
    """팀의 이번 분기 예산 현황(배정액, 사용액, 잔여액)을 조회한다."""
    return _budget_check(team_name, db_path=_db_path_from(config))


class PurchaseRequestInput(BaseModel):
    team_name: TeamName = Field(description="요청 팀명")
    product_id: str = Field(description="product_search로 조회한 product_id (예: 'P004')")
    quantity: int = Field(description="수량 (1 이상)")
    requester: str = Field(description="요청자 이름 또는 사번")


@tool("purchase_request", args_schema=PurchaseRequestInput)
def purchase_request_tool(
    team_name: str, product_id: str, quantity: int, requester: str, config: RunnableConfig
) -> dict:
    """구매 요청을 접수한다. 아직 예산에 실제로 반영되지는 않으며, purchase_register로 별도 집행해야 한다."""
    return _purchase_request(team_name, product_id, quantity, requester, db_path=_db_path_from(config))


class PurchaseRegisterInput(BaseModel):
    request_id: str = Field(description="purchase_request로 발급받은 요청 ID (예: 'REQ-0001')")


@tool("purchase_register", args_schema=PurchaseRegisterInput)
def purchase_register_tool(request_id: str, config: RunnableConfig) -> dict:
    """접수된 구매 요청을 실제로 집행해 팀 예산에 반영한다. 되돌릴 수 없는 최종 액션이므로 신중하게 호출해야 한다."""
    return _purchase_register(request_id, db_path=_db_path_from(config))


ALL_TOOLS = [
    product_search_tool,
    policy_search_tool,
    budget_check_tool,
    purchase_request_tool,
    purchase_register_tool,
]


if __name__ == "__main__":
    for t in ALL_TOOLS:
        print(f"- {t.name}: {t.description}")
        print(f"  args: {t.args}")