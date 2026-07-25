# Open Source Contribution Agent

一个采用 Claude Code 最小通用设计的 Python Agent：入口 Skill、子 Agent 和 Tool 全部复用同一个 `AgentRuntime`，完整 Session transcript 是唯一恢复依据。

## 架构

```text
Typer CLI / SkillTool
        ↓
SkillCommandRunner → SkillExecutor
        ↓
AgentRuntime.query
        ↓
ToolExecutor → Permission / Plan Mode → Hooks → Tool
        ↓
Session JSONL / Git Worktree Isolation
```

- Runtime 不知道 Contribution 阶段，也不存在固定 Workflow 状态机。
- `open-source-contribution` 是一个 inline 入口 Skill，按需读取四份阶段资源。
- Plan Mode 和 AskUserQuestion 提供方案审批与人工选择。
- Session、Plan、Tool Result 和 Worktree 保存在按仓库哈希隔离的用户数据目录，不污染目标仓库。
- Git worktree 提供真实执行隔离；脏 worktree 不会被静默删除。
- PowerShell Tool 明确调用 PowerShell 7 (`pwsh`)；本版本不提供 Bash 兼容入口。
- GitHub Tool 当前只读；不会自动 push、评论或创建远程 PR。

## 环境

```powershell
conda env create -f environment.yml
conda activate osc-agent
python -m pip install -e .
```

复制 `.env.example` 为 `.env` 并设置 `ANTHROPIC_API_KEY`。开发和验证优先使用 PowerShell 7。

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
- 新 Agent：添加 `AgentDefinition`，以 inline/fork 方式递归调用同一个 Runtime。
- 只有出现真实消费者时才增加 Task、MCP Transport 或更高层编排，不预建通用 Workflow DSL。

Skill 来源优先级为 `project > user > builtin`。

## 验证

```powershell
conda run -n osc-agent python -m pytest --basetemp .pytest-tmp-v2
conda run --no-capture-output -n osc-agent python -m osc_agent.cli --help
conda run --no-capture-output -n osc-agent python -m osc_agent.cli skill list --repo .
```
