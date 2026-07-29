"""定义跨能力层共享的严格 Pydantic 契约基类。"""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """跨模块传递且允许受控更新的严格契约。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class FrozenContractModel(BaseModel):
    """构建后不可变的严格契约。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )
