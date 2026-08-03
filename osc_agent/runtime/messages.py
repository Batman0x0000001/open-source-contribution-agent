"""定义 Provider、Runtime 和 Session 共用的消息协议。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue

from osc_agent.contracts import ContractModel


class TextBlock(ContractModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(ContractModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResultBlock(ContractModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1)
    content: JsonValue
    is_error: bool = False


MessageBlock: TypeAlias = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class RuntimeMessage(ContractModel):
    role: Literal["user", "assistant", "system"]
    content: list[MessageBlock] = Field(min_length=1)
