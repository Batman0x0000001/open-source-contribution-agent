# Bot-first architecture

本文描述 Open Source Contribution Agent `0.2.4` 的生产架构。

The GitHub App is the production product. The local CLI is a debugging adapter. Both build an
`AgentApplication`, open an `AgentConversation`, and submit typed input. The conversation facade
is the only product-facing caller of `AgentRuntime.query`.

```text
GitHub → Control → SQLite/Outbox → Worker ─┐
                                           ├→ AgentApplication → AgentConversation → AgentRuntime
Local debug CLI ───────────────────────────┘
```

Plan and implementation are deliberately separate Sessions. Plan uses `issue-planning`, only
read-only tools, Explore, and `DisabledProcessRunner`; it never resolves a Docker image.
Implementation uses the approved `open-source-contribution` Skill and a network-disabled Docker
runner. An immutable ExecutionContract binds the Issue snapshot, base SHA, model/profile/Skill
revisions, tools, validation, sandbox limits, repository policy, and publication mode.

These phases are trust boundaries rather than a task workflow. The Worker validates typed Job,
Plan, and ExecutionContract data, then constructs one `SkillInput`; the Skill controls the adaptive
method inside that Agent run. Discover, Design, Implement, and Verify are not Worker state-machine
steps, and Skill instructions cannot weaken sandbox, approval, artifact, or publication checks.

Worker 为 Plan 和 Implementation 建立独立 slot 配额。SIGTERM 会先停止领取新 Job，再有界
等待在途 Agent；超过关闭期限时取消任务，ProcessRunner 负责终止进程树或 Docker 容器。
RuntimeEvent 只按节流窗口保存事件类型和阶段。任何 denied path、Git metadata 或变更上限
违规都会立即把 Job 转为带 `REPOSITORY_POLICY_VIOLATION` 的不可发布终态。

## Control supervision

Uvicorn and OutboxDispatcher are peer critical tasks owned by ControlSupervisor. An unexpected
exit from either cancels the other and exits the process non-zero. Known external failures become
bounded retries or dead letters; database failures and unknown exceptions propagate. Readiness is
false whenever the dispatcher is stopped or its heartbeat is stale.

## `/osa reply`

A reply is stored exactly once as untrusted evidence, moves a blocked Plan back to the Plan queue,
and resumes the original Plan Session. The Plan profile remains read-only and cannot acquire
Docker, Process, write, approval, or implementation capabilities.

## Publication

Repositories explicitly select `draft` or `ready`. The current release does not accept an
`auto_merge` setting and never calls merge APIs.
