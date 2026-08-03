"""提供延迟读取大型工具结果的模型工具。"""

from __future__ import annotations

from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.runtime.session_store import ToolResultStore
from osc_agent.runtime.tool import BaseTool


class ReadToolResultInput(ContractModel):
    result_id: str = Field(
        min_length=1,
        description="Opaque result identifier returned by context compaction.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Character offset within the persisted result.",
    )
    limit: int = Field(
        default=4_000,
        ge=1,
        le=20_000,
        description="Maximum characters to return.",
    )


class ReadToolResultOutput(ContractModel):
    result_id: str
    content: str
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    complete: bool


class ReadToolResultTool(BaseTool[ReadToolResultInput, ReadToolResultOutput]):
    name = "read_tool_result"
    description = (
        "Read a bounded page of one large Tool result previously persisted for the "
        "current session. Continue from next_offset until complete is true."
    )
    input_model = ReadToolResultInput
    output_model = ReadToolResultOutput

    def __init__(self, store: ToolResultStore) -> None:
        self.store = store

    def is_read_only(self, input: ReadToolResultInput) -> bool:
        return True

    async def call(self, input: ReadToolResultInput, context: ToolContext) -> ToolResult:
        try:
            content = self.store.read(
                session_id=context.session_id,
                result_id=input.result_id,
            )
        except ValueError as exc:
            return ToolResult(
                error=ToolError(code="TOOL_RESULT_NOT_FOUND", message=str(exc))
            )
        page = content[input.offset : input.offset + input.limit]
        end = input.offset + len(page)
        complete = end >= len(content)
        return ToolResult(
            data={
                "result_id": input.result_id,
                "content": page,
                "offset": input.offset,
                "next_offset": None if complete else end,
                "complete": complete,
            }
        )
