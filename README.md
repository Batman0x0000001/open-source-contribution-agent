# Open Source Contribution Agent

一个采用 Claude Code 最小通用设计的 Python Agent：入口 Skill、子 Agent 和 Tool 全部复用同一个 `AgentRuntime`，完整 Session transcript 是唯一恢复依据。

## 架构

```text
Typer CLI / SkillTool
        ↓
SkillCommandRunner → SkillExecutor
        ↓
AgentRuntime.query
        ├── AgentTool → code-only AgentRegistry → AgentRunner
        ↓
ToolExecutor → Permission / Plan Mode → Hooks → Tool
        ↓
Session JSONL / Git Worktree Isolation
```

- Runtime 不知道 Contribution 阶段，也不存在固定 Workflow 状态机。
- `open-source-contribution` 是一个 inline 入口 Skill，按需读取四份阶段资源。
- Plan Mode 和 AskUserQuestion 提供方案审批与人工选择。
- 仓库内 `AGENTS.md` 与 `CLAUDE.md` 会按目录层级注入；同层任务相关冲突由 Agent 询问用户。
- `grep`、Read-before-write 文件观察和完整 Git snapshot 提供安全的代码理解与修改证据。
- Contribution Skill 结束前必须有最终修改之后的成功测试和更晚的 Git snapshot；无测试只能由用户显式豁免。
- Session、Plan、Tool Result 和 Worktree 保存在按仓库哈希隔离的用户数据目录，不污染目标仓库。
- Git worktree 提供真实执行隔离；脏 worktree 不会被静默删除。
- PowerShell Tool 明确调用 PowerShell 7 (`pwsh`)；本版本不提供 Bash 兼容入口。
- destructive permission 会显示 Tool、工作目录、风险和有界输入预览；PowerShell
  不允许直接执行 Git 写操作。
- GitHub Tool 当前只读；不会自动 push、评论或创建远程 PR。
- Contribution 可把两个独立的大型仓库调查问题交给只读 Explore 子 Agent；主 Agent
  只接收经过验证的证据报告。
- Contribution 在最终修改和主测试后调用只读 Verify 子 Agent；Verify 必须实际执行
  检查和对抗性探针，`FAIL` 不能豁免，`PARTIAL` 需要用户明确批准。
- 所有只读 Agent 均经过 Git workspace fingerprint 前后校验；如果 Agent 造成 Git
  可见变化，结果会被拒绝且现场不会被自动回滚。

## 环境

```powershell
conda env create -f environment.yml
conda activate osc-agent
python -m pip install -e .
```

复制 `.env.example` 为 `.env`，显式设置 `ANTHROPIC_API_KEY` 和 `MODEL_ID`。项目不内置
易过期的默认模型；使用自定义 `ANTHROPIC_BASE_URL` 时，`MODEL_ID` 应填写该网关接受的
模型标识。已存在的系统环境变量优先于 `.env`。开发和验证优先使用 PowerShell 7。

## CLI

```powershell
osc-agent run --repo C:\path\to\repo "分析并修复这个问题"
osc-agent contribute --repo C:\path\to\repo --repo-url https://github.com/org/repo
osc-agent resume --repo C:\path\to\repo <session-id>
osc-agent skill list --repo C:\path\to\repo
osc-agent skill run open-source-contribution --repo C:\path\to\repo --arguments '{"repo_url":"https://github.com/org/repo","goal":null}'
```

`run`、`contribute` 和 inline `skill run` 会先打印 Session ID。恢复不读取旧 Contribution Run 或阶段状态。

## 扩展

- 新 Tool：实现 `Tool` Protocol 并注册到 composition root。
- 新 Skill：添加严格 frontmatter；正文和声明的资源均延迟读取。
- 新内置 Agent：在代码中注册 `AgentRegistration`，并递归复用同一个 Runtime。
- 普通 `run` 与 Contribution 都可发现 AgentTool；当前代码注册的 Agent 是最多两个并行
  Explore 和一个串行 Verify。Agent 的 Tool、并发、预算和只读边界仍由 Registration 固定。
- Agent Registry 不扫描项目、用户、Plugin 或 Markdown Agent 文件。
- 只有出现真实消费者时才增加 Task、MCP Transport 或更高层编排，不预建通用 Workflow DSL。

Skill 来源优先级为 `project > user > builtin`。

## 验证

```powershell
conda run --no-capture-output -n osc-agent python -m pytest
conda run --no-capture-output -n osc-agent python -m osc_agent.cli --help
conda run --no-capture-output -n osc-agent python -m osc_agent.cli skill list --repo .
```

PowerShell 7 与 ripgrep 是源码运行环境的必需外部命令，`environment.yml` 会安装 ripgrep。
