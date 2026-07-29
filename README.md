# Open Source Contribution Agent

一个采用统一 Runtime 设计的 Python Agent：入口 Skill、子 Agent 和 Tool 全部复用同一个 `AgentRuntime`，完整 Session transcript 是唯一恢复依据。

> 当前项目与文档基线：`0.2.4`。

> 当前版本定位为 Ubuntu Bot 优先、本地 CLI 仅用于调试。Bash 在 Linux Host 上执行，
> 环境变量经过白名单过滤，但本地 CLI 没有 OS Sandbox；不要用它执行
> 恶意或不可信仓库中的命令。

`0.2.4` 提供可选的 GitHub App 服务端能力。它把 Webhook/发布凭据、Agent Worker 和无网络
Docker 仓库命令分成独立边界；该能力不会改变本地 CLI，也不会自动合并 PR。

## 架构

```text
Typer CLI / Bot Worker
        ↓
AgentApplicationConfig → AgentApplication.open_session
        ↓
AgentConversation.submit(UserPrompt | SkillInput | None)
        ↓
AgentRuntime.query ← SkillPreparer ← SkillTool
        ├── AgentTool → code-only AgentRegistry → AgentRunner
        ↓
ToolExecutor → Permission / Plan Mode → Hooks → Tool
        ↓
Session JSONL / Git Worktree Workspace
```

- Runtime 不知道 Contribution 阶段，也不存在固定 Workflow 状态机。
- `open-source-contribution` 是一个 inline 入口 Skill，按需读取四份阶段资源。
- Plan Mode 和 AskUserQuestion 提供方案审批与人工选择。
- 仓库内 `AGENTS.md` 与 `CLAUDE.md` 会按目录层级注入；同层任务相关冲突由 Agent 询问用户。
- `grep`、Read-before-write 文件观察和完整 Git snapshot 提供安全的代码理解与修改证据。
- Contribution Skill 结束前必须有最终修改之后的成功测试和更晚的 Git snapshot；无测试只能由用户显式豁免。
- Session、Plan、Tool Result 和 Worktree 保存在按仓库哈希隔离的用户数据目录，不污染目标仓库。
- Git worktree 提供独立工作目录和分支；脏 worktree 不会被静默删除。
- 进程 Tool 只提供非交互式 Bash；生产 Bot 在无网络 Linux 容器中执行，CLI 调试要求本机 Bash。
- Tool 子进程只接收运行必需环境变量和用户显式允许的非敏感变量；API Key、Token、
  Secret、Password 等凭据名称始终被过滤。
- destructive permission 会显示 Tool、工作目录、风险和有界输入预览；Bash
  不允许直接执行 Git 写操作。
- `write`/`process` 权限可选择仅本次或当前 Session；Session 记忆只匹配完全相同的
  Tool、风险、工作目录和规范化输入，destructive/external 永不缓存。
- 所有模型消费者共享有界指数退避重试；失败的部分流不会进入 transcript 或触发 Tool。
- GitHub Tool 当前只读；不会自动 push、评论或创建远程 PR。
- Contribution 可把两个独立的大型仓库调查问题交给只读 Explore 子 Agent；主 Agent
  只接收经过验证的证据报告。
- Contribution 在最终修改和主测试后调用只读 Verify 子 Agent；Verify 必须实际执行
  检查和对抗性探针，`FAIL` 不能豁免，`PARTIAL` 需要用户明确批准。
- 所有只读 Agent 均经过 Git workspace fingerprint 前后校验；如果 Agent 造成 Git
  可见变化，结果会被拒绝且现场不会被自动回滚。

## 环境

```bash
conda env create -f environment.yml
conda activate osc-agent
python -m pip install -e .
```

复制 `.env.example` 为 `.env`，显式设置 `ANTHROPIC_API_KEY` 和 `MODEL_ID`。项目不内置
易过期的默认模型；使用自定义 `ANTHROPIC_BASE_URL` 时，`MODEL_ID` 应填写该网关接受的
模型标识。已存在的系统环境变量优先于 `.env`。开发、验证和部署统一使用 Linux Bash。

## CLI

```bash
osc-agent run --repo /path/to/repo "分析并修复这个问题"
osc-agent run --repo /path/to/repo "分析并修复这个问题" --once
osc-agent contribute --repo /path/to/repo --repo-url https://github.com/org/repo
osc-agent resume --repo /path/to/repo <session-id>
osc-agent resume --repo /path/to/repo --latest
osc-agent session list --repo /path/to/repo
osc-agent session show <session-id> --repo /path/to/repo
osc-agent doctor --repo /path/to/repo
osc-agent doctor --repo /path/to/repo --local-only
osc-agent skill list --repo /path/to/repo
osc-agent skill run open-source-contribution --repo /path/to/repo --arguments '{"repo_url":"https://github.com/org/repo","goal":null}'
```

`run`、`resume`、`contribute` 和 inline `skill run` 会先打印 Session ID。在交互式
终端中，它们默认在每轮结束后继续接收同一 Session 的下一条消息；使用 `--once`
保持脚本化单轮行为。输入 `/exit`、Ctrl+C（位于输入提示）或 EOF 可退出。

CLI 默认把模型轮次、Tool、Agent、重试和 compact 状态以紧凑行写入 stderr，并在每轮
结束后根据类型化 Tool Result 显示确定性摘要。`--quiet` 只隐藏实时状态，不隐藏摘要。

`doctor` 默认发送一次最小模型请求验证真实连接，因此会消耗极少量 Token；
`--local-only` 只执行配置、Bash、ripgrep、Git、状态目录、Skill 和 Agent 检查。
模型请求默认最多尝试三次，可通过 `.env.example` 中的重试变量调整。

## 扩展

- 新 Tool：实现 `Tool` Protocol 并通过 `AgentApplicationConfig` 注册到 composition root。
- 新 Skill：添加严格 frontmatter；正文和声明的资源均延迟读取。
- 新内置子 Agent：在代码中注册 `SubagentRegistration`，并递归复用同一个 Runtime。
- 普通 `run` 与 Contribution 都可发现 AgentTool；当前代码注册的 Agent 是最多两个并行
  Explore 和一个串行 Verify。Agent 的 Tool、并发、预算和只读边界仍由 Registration 固定。
- Agent Registry 不扫描项目、用户、Plugin 或 Markdown Agent 文件。
- 只有出现真实消费者时才增加 Task、MCP Transport 或更高层编排，不预建通用 Workflow DSL。

内置 Skill 名称是保留名称，项目或用户 Skill 不能覆盖。模型只能发现和调用内置 Skill；
非内置 Skill 必须由用户通过精确的 `osc-agent skill run <name>` 显式调用。对于非保留的
同名 Skill，显式解析仍采用 `project > user`。

## Host 执行边界

- 自动无需批准的 Bash 命令仅限受约束的 `rg` 和只读 Git 子命令；其他命令需要
  人工批准，提示会明确说明它运行在 Host 且没有 OS Sandbox。
- 提权、挂载、服务管理、关机、根目录递归删除和 fork bomb 等危险 Bash 命令被硬拒绝，
  批准也不能绕过。
- `OSC_AGENT_SUBPROCESS_ENV_ALLOWLIST` 接受 JSON 字符串数组，只用于额外放行非敏感
  变量名；受保护凭据永远不会传入 Tool 子进程。
- GitHub Issue、评论、仓库指令和 Tool 输出只作为证据，不能授予 Capability、Permission
  或绕过 Plan Mode。

## GitHub App Bot（可选）

```bash
python -m pip install -e ".[bot]"
osc-agent-bot doctor --control
osc-agent-bot doctor --worker
osc-agent-bot control
osc-agent-bot worker
```

Bot 是生产主入口；CLI 保留为本地调试入口。机器人只接受具有 `write`、`maintain` 或
`admin` 权限用户在 Issue 下的精确命令：

```text
/osa plan
/osa implement
/osa reply <补充信息>
/osa status
/osa retry
/osa cancel
```

每个 Issue 同时只允许一个活动 Job。Plan 是无 Docker 的只读 Session；Implementation 使用
全新的 Session。每个仓库必须显式配置完整的 `sha256:<64 hex>` 不可变执行镜像 ID、至少
一条测试命令和 `pull_request_mode: draft|ready`。当前版本不支持 auto-merge。
Control 不访问 Docker；Worker 在运行 Job 前验证本机镜像与该 ID 精确一致。Plan Session 永久只读；
Implementation 使用全新的 Session 和 clone。模型与 API Key 留在 Worker 宿主进程，项目
命令在 `--network none`、只读 root filesystem、只读 `.git` 的临时 Docker 容器中运行。
只有 Control/Publisher 能读取 GitHub App 私钥、commit、push 和创建 Draft PR。

Ubuntu 双用户、systemd、Nginx、权限与云服务器安全组配置见
`deploy/ubuntu/README.md`。这部分需要独立部署授权；源码不会注册 GitHub App 或修改服务器。

## 验证

```bash
conda run --no-capture-output -n osc-agent python -m pytest
conda run --no-capture-output -n osc-agent python -m osc_agent.cli --help
conda run --no-capture-output -n osc-agent python -m osc_agent.cli skill list --repo .
conda run --no-capture-output -n osc-agent python -m osc_agent.bot --help
```

Bash 与 ripgrep 是源码运行环境的必需外部命令，`environment.yml` 会安装 ripgrep。
