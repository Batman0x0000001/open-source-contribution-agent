# Agent Runtime 与代码阅读地图

本文描述统一 Agent 执行内核及其纵向入口。目录划分服务于阅读路径，而不是文件数量。

## 总体数据流

```text
CLI / Bot Worker
    → AgentApplicationConfig
    → build_agent_application()
    → AgentApplication.open_session()
    → AgentConversation.start() / resume()
    → AgentRuntime.query()
    → AgentRunState
    → ToolContext
    → ToolResult.state_changes
    → AgentRunState.apply()
    → Session V6
```

`application/agent.py` 是唯一 composition root，也是产品进入 Runtime 的唯一入口。CLI 与
Bot Worker 都不能构造 `StartQueryParams`、`ResumeQueryParams` 或直接调用 Runtime。

## 按目标阅读

```text
想看 CLI 调用       → cli/app.py → cli/agent.py
想看 CLI Session    → cli/sessions.py
想看 Bot Agent Run → bot/worker/coordinator.py → bot/worker/agent_jobs.py
想看应用组装        → application/agent.py
想看模型循环        → runtime/query.py
想看运行状态        → runtime/state.py
想看 Tool 执行      → runtime/tool_execution.py → runtime/tool_orchestration.py
想看具体 Tool       → tools/<domain>.py
想看完成判定        → completion/evaluator.py
想看可信发布        → bot/control/publisher.py
```

## Application

`application/agent.py` 从上到下依次包含产品配置、Application、Conversation 和组装函数。
Application 直接持有 Conversation 所需的 Runtime、SessionStore、SkillPreparer、QueryConfig 和
有效 capabilities，不存在中间 `ApplicationGraph`。CLI 与 Bot Worker 显式选择 Start 或
Resume；Conversation 只翻译输入，Runtime 在 Session lease 内验证创建或恢复条件。

Skill 的默认来源由 `skills/catalog.py` 构建，Provider 由 `providers/factory.py` 构建；doctor
和只读查询不需要构造完整 Application。

## Runtime 与状态

`AgentRuntime.query()` 是唯一模型循环。不可变 Query 参数、私有执行指标 `_QueryProgress` 和
可持久化 `AgentRunState` 分离：前者只控制预算、压缩恢复、无进展与 Completion Gate 计数，
后者保存 workspace、permissions、capabilities、completion requirements 与唯一终态。

Tool 只接收从状态投影出的只读 `ToolContext`。Tool 不修改上下文，而是返回有序
`state_changes`。串行调用立即应用变化；并发调用可乱序完成，但变化始终按模型调用顺序应用。
Tool Result 到达和 Batch State Commit 使用不同事件表达；并发 Result 可以先到达，但只有
Commit 事件能够替换并持久化 `AgentRunState`。Tool 已执行后若 PostToolUse Hook 失败，Runtime
保留真实 Result 与 StateChange，再将 Query 终止为失败，避免模型重试已经发生的副作用。
capability 和 completion requirement 只能收窄，合并规则只存在于 `AgentRunState.apply()`。
Tool Schema 根据该状态展示能力，ToolExecutor 在 PermissionPolicy 之前执行相同 capability
硬门禁。Skill 与 Agent 的发现信息只存在于对应 Tool description，不复制进 system prompt。

Context projection 只改变发送给模型的视图，不能改变权威 transcript 或破坏
`tool_use/tool_result` 配对。Reactive Compact、取消配对、预算和完成阻断仍由唯一 Query
循环处理。

## Session V6

V6 Metadata 只保存不可变身份：`session_id`、`workspace_root`、`model` 和 `system_prompt`。
当前 workspace、permissions、capabilities、completion requirements 与终态只保存在
`AgentRunState`，不存在平行字段。File 与 SQLite Store 使用相同模型；V5 明确拒绝，不迁移。
恢复始终使用持久化模型、系统提示和运行状态；当前 Profile 不能覆盖旧 Session。

## Completion

`completion/evaluator.py` 是与 Runtime 无关的中立能力。Runtime 的 `CompletionStopHook` 只把
Hook 输入适配为 `CompletionEvaluation`；Bot Publisher 直接调用同一个 Evaluator。相同的
transcript、workspace、requirements 和 validation commands 必须产生相同 blocking reasons。

## Tool、Workspace 与 Process

`tools/` 只保存模型 Tool 协议和薄适配；文件、Git 与路径能力在 `workspaces/`，进程契约和
Host runner 在 `processes/`。`ProcessRequest.invocation_id` 提供容器命名信息，ProcessRunner
不依赖 Runtime Context。Application 是唯一允许同时导入 Provider、具体 Tools、Skills、
Subagents、Runtime 和 Workspace 实现的模块。

## Skill 与 Subagent

Skill 是当前 Conversation 的声明式方法，不拥有模型循环。隔离调查和验证统一走
`Runtime → AgentTool → SubagentRunner → 同一 Runtime`；子 Agent 使用独立 Session，并移除
`agent` capability 防止递归。

## Bot 信任边界

Control 处理 Webhook、审批、Outbox 与可信发布，不进入 Agent Runtime。Worker 的
`agent_jobs.py` 顺序包含 Plan 与 Implementation 两条完整用例；两者仍是必要的信任阶段，
不是通用 Workflow。Plan 使用 Disabled runner，Implementation 使用无网络 Docker runner。
