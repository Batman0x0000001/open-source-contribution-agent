# Agent Runtime 核心架构

## 目的

本文档定义 Open Source Contribution Agent `0.2.4` 的通用 Agent 核心，以及 Runtime、Tool、Context、Skill、Agent 和 Session 之间的稳定职责边界。

## 核心数据流

```text
AgentApplicationConfig
          │
          ▼
AgentApplication.open_session(session_id)
          │
          ▼
AgentConversation.submit(UserPrompt | SkillInput | None)
          │
          ▼
QueryConfig + QueryDependencies + QueryState
                    │
                    ▼
             AgentRuntime.query
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   ModelGateway          Context Projection
         │
         ▼
     Tool requests
         │
         ▼
   ToolOrchestration
         │
         ▼
 Validation → Permission → Hooks → Tool.call
         │
         ▼
 ToolResult + ContextUpdate + RuntimeEvent
         │
         └──────────────► 下一轮 QueryState
```

## Query

Query 是异步生成器，而不是只返回最终文本的同步函数。它负责推进一次 Agent turn，并持续输出模型流、Tool 事件、上下文事件和终态。

Query 内部必须分离：

- 不可变的 `QueryConfig`。
- 可注入的 `QueryDependencies`。
- 跨轮变化的 `QueryState`。
- 受控的 `ToolUseContext`。

Query 是唯一允许推进 QueryState 的组件。CLI、Skill 和 Tool 都不能直接修改它。

`repository_root`、Profile 和 Runtime 依赖只在构建 `AgentApplication` 时绑定。产品入口只持有
`AgentConversation`：新 Session 必须提交 Prompt 或 Skill，恢复 Session 可提交新 Prompt 或
`None`。`StartQueryParams` 与 `ResumeQueryParams` 是 application 包到 Runtime 的内部协议。

## Tool

Tool 是完整行为对象，同时包含：

- 输入和输出 Pydantic 模型。
- 启用判断。
- 按输入计算的只读与并发安全判断。
- 输入验证和 Tool 专属权限检查。
- 执行方法。
- 最大结果大小。

所有 Tool 都经过同一执行管线。并发安全的连续调用组成并发批次，其他调用串行执行。并发完成顺序可以变化，但 ContextUpdate 必须按原 Tool call 顺序应用。

## Context

完整 Transcript 是权威历史；Context Projection 是当前模型请求使用的视图。上下文压缩只能改变 Projection，不能破坏 Transcript 或 `tool_use/tool_result` 配对。

最小管线包括：

1. 超大 Tool Result 落盘并生成预览。
2. 压缩旧 Tool Result。
3. 接近窗口限制时 Auto Compact。
4. API 拒绝上下文时执行有上限的 Reactive Compact。

## Skill

Skill 是延迟加载的声明式 Agent 方法。Catalog 默认只读取发现元数据，正文在调用时加载；
`when_to_use` 同时进入 CLI 和模型发现信息。模型侧的 `SkillTool` 和产品侧的
`AgentConversation.submit(SkillInput)` 共用 `SkillPreparer`，后者只做调用授权、自由 JSON
参数渲染和 capability 收窄。用户与模型调用由 Catalog 策略控制；产品主动启动还必须由
`AgentProfile.allowed_initial_skills` 显式授权，且只有该路径能获得 manifest 的
`product_tools`。Manifest 声明的资源由 `read_skill_resource` 按需读取，并受解析后根目录
边界保护。

Skill 不拥有子模型循环。隔离调查和验证统一通过 `AgentTool → SubagentRunner → 同一
AgentRuntime.query()` 执行。推荐阅读顺序是：`models → loader → catalog → preparer →
invocation_tool → resource_tool → builtins`。

## Agent

`AgentTool` 委派给 `SubagentRunner`，后者通过隔离的 `StartQueryParams` 和 QueryState
递归调用同一个 `AgentRuntime.query()`。子 Agent 的 capability 必须显式枚举，并在运行前
移除 `agent` 工具以禁止递归。main agent 和 minimal/fork subagent 不允许拥有第二套模型
循环。后台 Agent 在最小稳定版中不提供。

## Session、Plan 与 Contribution

完整 JSONL Session transcript 是恢复依据；Plan Mode 与 Worktree 是少量类型化会话状态。
`workspaces/git_worktree.py` 管理 Git 工作目录生命周期，`tools/worktree.py` 只负责权限、
输入输出与 Session Context 适配。Contribution 是 inline Skill，不是 Workflow：Agent Loop
根据 transcript、批准的 plan、测试和 git diff 动态推进，人工检查点由 AskUserQuestion 与
PermissionPolicy 表达。

## Pydantic 与 Protocol

- 数据、状态、事件、输入输出和 Artifact 使用严格 Pydantic 模型。
- Tool、ModelGateway、Store、Executor 等行为边界使用 `Protocol`。
- 包含函数和服务对象的依赖容器使用冻结 `dataclass`。
- 未验证的 SDK 数据只能存在于 Provider adapter 内部。

## 扩展边界

后续扩展只依赖四个稳定入口：

- 注册 Tool 增加原子能力。
- 注册 Skill 增加知识和任务方法。
- 通过 `AgentApplicationConfig` 绑定产品配置。
- 通过 `AgentConversation`、Skill 与 Tool 组合新的用户任务入口。

在出现真实分发需求前，不引入额外 Plugin 框架。
