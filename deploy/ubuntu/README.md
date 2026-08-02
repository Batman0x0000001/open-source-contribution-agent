# Ubuntu 单机部署

本文适用于 Open Source Contribution Agent `0.3.0`。

推荐从源码发布目录运行 `sudo ./deploy/ubuntu/install.sh --config /etc/osc-agent/config.yml`，
再分别执行 Control/Worker doctor。升级采用停机归档和当前状态模型的原子重建，不支持不同
状态模型混跑。Prometheus 告警模板见 `prometheus-alerts.yml`。

当前部署面向一台 Ubuntu 主机和 SQLite WAL。Control 与 Worker 必须使用不同 OS 用户：Control
持有 GitHub App 私钥但没有 Docker 权限；Worker 可调用 Docker，但其环境文件不包含任何
GitHub App 私钥、Webhook Secret 或 Publisher 身份。

## 目录与用户

```bash
sudo groupadd --system osa-shared
sudo useradd --system --home /var/lib/osc-agent --gid osa-shared osa-control
sudo useradd --system --home /var/lib/osc-agent --gid osa-shared osa-worker
sudo usermod -aG docker osa-worker
sudo install -d -m 2770 -o osa-control -g osa-shared /var/lib/osc-agent
sudo install -d -m 2770 -o osa-control -g osa-shared /var/lib/osc-agent/workspaces
sudo install -d -m 0750 -o root -g osa-shared /etc/osc-agent
```

安装 Docker Engine、Git、Bash、ripgrep、Nginx 和项目固定版本 bot wheel。安装器将 wheel
及完整依赖集冻结到 `/opt/osc-agent/releases/<version>/wheels`，并原子切换 `current`。预先构建
仓库配置引用的镜像，并将 `docker image inspect --format '{{.Id}}' <tag>` 的完整结果写入
`config.yml` 的 `repositories` 区段；tag、短 digest 和大写 digest 都会被拒绝。镜像内必须有
`bash` 和项目验证依赖。不要把 Docker socket、数据库、
状态根目录或 Secret 挂入容器。

复制示例配置：

- `/etc/osc-agent/bot.env`：owner `osa-control:osa-shared`，mode `0600`，仅 Control。
- `/etc/osc-agent/worker.env`：owner `osa-worker:osa-shared`，mode `0600`，仅 Worker。
- 将 `config.example.yml` 复制为 `/etc/osc-agent/config.yml`：owner `root:osa-shared`，
  mode `0640`。它是唯一非敏感生产配置，保存 Agent 预算、模型重试和仓库执行契约输入；
  Secret 仍只存在于两个 mode `0600` 的 EnvironmentFile。

Control 与 Worker 的 primary group 都是 `osa-shared`，systemd `UMask=0007`，因此 Control
创建的新鲜 clone 可由 Worker 及其同 UID/GID 的非 root 容器写入。不要以 root 运行 Worker。

## GitHub App

Webhook URL 为 `https://<public-host>/webhooks/github`。只订阅
`issue_comment`，最小仓库权限为 Metadata read、Issues write、Pull requests write、Contents
write；不要授予 Workflows 权限。部署后分别通过使用对应 `EnvironmentFile` 的临时
systemd unit 或受控维护终端运行 `osc-agent-bot doctor --control` 和
`osc-agent-bot doctor --worker`。Control Doctor 不访问 Docker；Worker Doctor 校验
`git`、`docker`、`bash`、`rg`、模型配置和本机镜像。不要用命令替换把 env 文件展开到 argv
或 shell history；Doctor 本身不会输出凭据值。

## 服务与网络

将本目录的 unit 文件复制到 `/etc/systemd/system/`，Nginx 配置复制到站点配置后启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now osc-agent-bot-control osc-agent-bot-worker osc-agent-bot-cleanup.timer
sudo nginx -t
sudo systemctl reload nginx
```

升级维护标记固定为 `/run/osc-agent-webhook-maintenance`。不要把它放到权限为 `0750`
的 `/etc/osc-agent`；非特权 Nginx Worker 无法检查该目录中的文件，维护开关会失效。

云服务器安全组只公开 HTTPS 443；SSH 22 仅允许管理来源 IP。8080 仅监听
`127.0.0.1`。SQLite、Docker API 和任何内部端口不得公开。Control 需要访问 GitHub API，
Worker 宿主进程需要访问模型 API；Docker 仓库命令始终完全断网。

Worker 分别使用 `OSC_AGENT_BOT_MAX_CONCURRENT_PLANS` 和
`OSC_AGENT_BOT_MAX_CONCURRENT_IMPLEMENTATIONS` 控制两个阶段；收到 SIGTERM 后停止领取新
Job，并在 `OSC_AGENT_BOT_SHUTDOWN_TIMEOUT_SECONDS` 内等待在途任务清理，超时后取消。

日志交给 journald，并由系统策略轮转。完成 workspace 默认 24 小时后清理，Job/审计默认
30 天后清理。备份 SQLite 时同时处理 `-wal`/`-shm`，或先停止两个服务并使用 SQLite
在线备份机制。

## 运维命令

```bash
osc-agent-bot doctor --control
osc-agent-bot doctor --worker
osc-agent-bot smoke-test
osc-agent-bot archive-state
osc-agent-bot reset-state --confirm
```

当前状态模型的升级不迁移运行中的 Job。`upgrade.sh` 先让 Webhook 返回 503，同时停止 Control 与
Worker，持有升级锁并归档 SQLite/WAL/SHM/workspace；随后原子切换 release symlink、重建
当前数据库，按 Control→Worker 顺序启动，通过 smoke 后才恢复 Webhook。归档状态仅用于
归档审计：`waiting_implementation → waiting_approval`、`blocked → blocked_plan`、
`failed → dead_letter`，不会作为可恢复 Job 导入新库。
新建 SQLite 使用共享组可写的 `0660`，workspace 根目录使用 `2770`，因此 Control 与 Worker
无需共享 OS 用户也能安全协作。
