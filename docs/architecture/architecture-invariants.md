# 项目架构不变量矩阵

本文记录 Open Source Contribution Agent `0.3.0` 由自动化测试保护的核心设计约束。

| 编号 | 最小机制 | Python 对应 | 硬验证 |
|---|---|---|---|
| Q1 | Query 是唯一 AsyncGenerator 循环 | `AgentRuntime.query` | 事件序列与单一入口测试 |
| Q2 | 配置、循环计数、持久状态、Tool 投影分离 | `QueryConfig`、私有 `_QueryState`、`AgentRunState`、`ToolContext` | Pydantic / frozen dataclass |
| T1 | Tool 是输入敏感的完整行为对象 | `Tool`、`ToolExecutor` | validation → permission → hooks → call 测试 |
| T2 | 并发完成可乱序，StateChange 按调用顺序应用 | `tool_orchestration.py`、`AgentRunState.apply()` | 并发与串行测试 |
| T3 | Tool 只适配模型协议，共享文件、Git 和进程能力位于 Tool 外部 | `tools/`、`workspaces/`、`processes/` | AST 依赖边界与核心注册表测试 |
| C1 | Transcript 与模型 Projection 分离 | `SessionTranscript`、`ContextPipeline` | compact 不改变权威历史和 Tool 配对 |
| R1 | Session V6 无平行运行字段，Transcript 驱动恢复 | `SessionMetadata`、`AgentRunState`、File/SQLite Store | V5 拒绝、V6 新建、追加与恢复 |
| S1 | Catalog 只发现 manifest，正文和资源延迟加载 | `SkillLoader`、`ReadSkillResourceTool` | 调用时读取与路径逃逸测试 |
| S2 | 产品 Skill、SkillTool 共用准备器 | `AgentConversation`、`SkillTool`、`SkillPreparer` | composition 对象同一性测试 |
| S3 | Skill 只注入当前 Conversation，不创建子模型循环 | `SkillPreparer`、`AgentTool` | Skills 禁止导入 Subagents 与旧 fork 路径测试 |
| S4 | 产品主动启动 Skill 必须由 Profile 显式授权 | `allowed_initial_skills`、`product_tools` | 隐藏 Skill、CLI 与 Bot 入口授权测试 |
| P1 | Plan Mode 是 Permission 状态，不是 Workflow Gate | plan tools、`DefaultPermissionPolicy` | 禁止普通写、固定计划路径、批准退出 |
| A1 | 子 Agent 递归复用 Query，且不能再次调用 AgentTool | `SubagentRunner`、`AgentTool` | Runtime 同一性、隔离与 capability 收窄测试 |
| W1 | Git worktree 是工作区隔离 | `GitWorktreeManager`、enter/exit tools | 创建、上下文切换、脏状态保护 |
| D1 | 无第二套执行架构 | `runtime/`、`subagents/`、`skills/`、`workspaces/` | 禁止 `support`、`workflows`、Todo、Task、Mock MCP 依赖 |
| B1 | 所有产品入口共享 Agent 生命周期 | `AgentApplication`、`AgentConversation` | CLI/Worker 不直接构造 Query 参数 |
| B3 | 仓库与 Profile 只在构建时绑定 | `AgentApplicationConfig` | 单轮输入不能覆盖仓库路径或 Profile |
| B2 | 服务状态机不侵入 Agent Runtime | `BotJobStateMachine` | Runtime 不 import Bot，状态图自动校验 |
| B4 | 产品入口显式选择 Start/Resume，Runtime 在 lease 内验证 Session 事实 | `AgentConversation`、`AgentRuntime` | CLI 严格恢复与 Bot crash-safe 分发测试 |
| E1 | Runtime 与 Publisher 使用同一完成判定 | `CompletionEvaluator`、`CompletionStopHook` | 相同输入得到相同 blocking reasons |
| X1 | Workspace、Process、Bot Domain/Control 不依赖 Runtime Context | 包依赖方向 | AST 边界测试 |
| X2 | Capability 展示与执行使用同一持久状态 | `AgentRunState`、`ToolRegistry`、`ToolExecutor` | Schema 过滤与不可绕过执行门禁测试 |

Bot 只持有 Plan/Approval/Implementation/Publish 的信任转换；分析、规划、实现和验证方法仍由
Skill 驱动。Skill Prompt 不能替代 ExecutionContract、Docker、Artifact Tool 或 Publisher
的强制校验。

新增 Runtime 抽象必须对应当前消费者和可验证机制；无法映射到核心机制的预留框架不进入 Runtime。
