# 项目架构不变量矩阵

本文记录 Open Source Contribution Agent `0.2.4` 由自动化测试保护的核心设计约束。

| 编号 | 最小机制 | Python 对应 | 硬验证 |
|---|---|---|---|
| Q1 | Query 是唯一 AsyncGenerator 循环 | `AgentRuntime.query` | 事件序列与单一入口测试 |
| Q2 | 配置、依赖、状态、Tool 上下文分离 | `QueryConfig`、`QueryDependencies`、`QueryState`、`ToolUseContext` | Pydantic / frozen dataclass |
| T1 | Tool 是输入敏感的完整行为对象 | `Tool`、`ToolExecutor` | validation → permission → hooks → call 测试 |
| T2 | 并发完成可乱序，ContextUpdate 按调用顺序应用 | `tool_orchestration.py` | 并发与串行测试 |
| C1 | Transcript 与模型 Projection 分离 | `SessionTranscript`、`ContextPipeline` | compact 不改变权威历史和 Tool 配对 |
| R1 | Transcript 驱动恢复 | `FileSessionStore` | JSONL 新建、追加、损坏拒绝、状态恢复 |
| S1 | Catalog 只发现 manifest，正文和资源延迟加载 | `SkillLoader`、`ReadSkillResourceTool` | 调用时读取与路径逃逸测试 |
| S2 | 产品 Skill、SkillTool 共用执行器 | `AgentConversation`、`SkillTool`、`SkillExecutor` | composition 对象同一性测试 |
| P1 | Plan Mode 是 Permission 状态，不是 Workflow Gate | interaction tools、`DefaultPermissionPolicy` | 禁止普通写、固定计划路径、批准退出 |
| A1 | 子 Agent 递归复用 Query | `AgentRunner`、`AgentTool` | Runtime 同一性与隔离测试 |
| W1 | Git worktree 是执行隔离 | `WorktreeManager`、enter/exit tools | 创建、上下文切换、脏状态保护 |
| D1 | 无第二套执行架构 | `runtime/`、`agents/`、`skills/`、`isolation/` | 禁止 `support`、`workflows`、Todo、Task、Mock MCP 依赖 |
| B1 | 所有产品入口共享 Agent 生命周期 | `AgentApplication`、`AgentConversation` | CLI/Worker 不直接构造 Query 参数 |
| B3 | 仓库与 Profile 只在构建时绑定 | `AgentApplicationConfig` | 单轮输入不能覆盖仓库路径或 Profile |
| B2 | 服务状态机不侵入 Agent Runtime | `BotJobStateMachine` | Runtime 不 import Bot，状态图自动校验 |

新增 Runtime 抽象必须对应当前消费者和可验证机制；无法映射到核心机制的预留框架不进入 Runtime。
