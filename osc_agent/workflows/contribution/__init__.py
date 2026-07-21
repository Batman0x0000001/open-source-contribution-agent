"""开源贡献工作流的稳定公共 API。"""

from osc_agent.workflows.contribution.gates import (
    GateResult,
    gate_design,
    gate_discover,
    gate_implementation,
)
from osc_agent.workflows.contribution.models import ContributionRun
from osc_agent.workflows.contribution.design import (
    attach_design_agent_review,
    build_design_review_prompt,
    configure_run,
    design_stage,
    render_design,
    update_design_contract,
    validate_design_files,
)
from osc_agent.workflows.contribution.discover import (
    attach_discover_human_review,
    build_discover_evidence,
    build_discover_review_prompt,
    discover_stage,
    render_discover,
    revalidate_selected_issue,
)
from osc_agent.workflows.contribution.implementation import (
    execute_implementation_stage,
    implement_stage,
    prepare_implementation_stage,
    record_implementation_result,
    record_test_waiver,
    render_implementation_report,
    run_verification_commands,
    validate_implementation_scope,
)
from osc_agent.workflows.contribution.prompts import (
    build_edit_prompt,
    build_implementation_prompt,
    build_reproduction_prompt,
    build_repair_prompt,
    build_understanding_prompt,
    build_verification_prompt,
    implementation_prompt_for_run,
)
from osc_agent.workflows.contribution.pr_draft import build_workflow_pr_draft, draft_pr_stage
from osc_agent.workflows.contribution.state import bind_run_worktree, create_run, load_run, save_run
from osc_agent.workflows.contribution.transitions import transition_run

__all__ = [
    "ContributionRun",
    "GateResult",
    "attach_design_agent_review",
    "attach_discover_human_review",
    "bind_run_worktree",
    "build_design_review_prompt",
    "build_discover_evidence",
    "build_discover_review_prompt",
    "build_edit_prompt",
    "build_implementation_prompt",
    "build_reproduction_prompt",
    "build_repair_prompt",
    "build_understanding_prompt",
    "build_verification_prompt",
    "build_workflow_pr_draft",
    "configure_run",
    "create_run",
    "design_stage",
    "discover_stage",
    "draft_pr_stage",
    "execute_implementation_stage",
    "gate_design",
    "gate_discover",
    "gate_implementation",
    "implement_stage",
    "implementation_prompt_for_run",
    "load_run",
    "prepare_implementation_stage",
    "record_implementation_result",
    "record_test_waiver",
    "render_design",
    "render_discover",
    "render_implementation_report",
    "revalidate_selected_issue",
    "run_verification_commands",
    "save_run",
    "transition_run",
    "update_design_contract",
    "validate_design_files",
    "validate_implementation_scope",
]
