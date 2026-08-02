"""提供延迟读取大型工具结果的模型工具。"""

from __future__ import annotations

from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.runtime.session_store import ToolResultStore
from osc_agent.runtime.tool import BaseTool


class ReadToolResultInput(ContractModel):
    result_id: str = Field(min_length=1, description="Opaque result identifier returned by context compaction.")


class ReadToolResultOutput(ContractModel):
    result_id: str
    content: str


class ReadToolResultTool(BaseTool[ReadToolResultInput, ReadToolResultOutput]):
    name = "read_tool_result"
    description = "Read one large Tool result previously persisted for the current session."
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
        return ToolResult(data={"result_id": input.result_id, "content": content})
