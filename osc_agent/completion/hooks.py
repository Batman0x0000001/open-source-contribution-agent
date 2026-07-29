"""将中立 CompletionEvaluator 适配为 Runtime StopHook。"""

from osc_agent.completion.evaluator import CompletionEvaluation, CompletionEvaluator
from osc_agent.runtime.hooks import StopHookPayload, StopHookResult
from osc_agent.runtime.state import ToolContext


class CompletionStopHook:
    def __init__(
        self,
        evaluator: CompletionEvaluator | None = None,
        *,
        validation_commands: tuple[str, ...] = (),
    ) -> None:
        self.evaluator = evaluator or CompletionEvaluator()
        self.validation_commands = validation_commands

    async def __call__(
        self,
        payload: StopHookPayload,
        context: ToolContext,
    ) -> StopHookResult:
        report = await self.evaluator.evaluate(
            CompletionEvaluation(
                messages=tuple(payload.messages),
                workspace_root=context.workspace.working_directory,
                requirements=context.completion_requirements,
                validation_commands=self.validation_commands,
            )
        )
        return StopHookResult(blocking_reasons=list(report.blocking_reasons))
