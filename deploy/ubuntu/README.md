# Ubuntu 单机部署

V9 面向一台 Ubuntu 主机和 SQLite WAL。Control 与 Worker 必须使用不同 OS 用户：Control
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

安装 Docker Engine、Git、PowerShell 7、ripgrep、Nginx 和项目 editable bot extra。预先构建
仓库配置引用的镜像，并将 `docker image inspect --format '{{.Id}}' <tag>` 的完整结果写入
`repositories.yml` 的 `image`；tag、短 digest 和大写 digest 都会被拒绝。镜像内必须有
`pwsh` 和项目验证依赖。不要把 Docker socket、数据库、
状态根目录或 Secret 挂入容器。

复制示例配置：

- `/etc/osc-agent/bot.env`：owner `osa-control:osa-shared`，mode `0600`，仅 Control。
- `/etc/osc-agent/worker.env`：owner `osa-worker:osa-shared`，mode `0600`，仅 Worker。
- `/etc/osc-agent/repositories.yml`：owner `root:osa-shared`，mode `0640`。

Control 与 Worker 的 primary group 都是 `osa-shared`，systemd `UMask=0007`，因此 Control
创建的新鲜 clone 可由 Worker 及其同 UID/GID 的非 root 容器写入。不要以 root 运行 Worker。

## GitHub App

Webhook URL 为 `https://<public-host>/webhooks/github`。只订阅
`issue_comment`，最小仓库权限为 Metadata read、Issues write、Pull requests write、Contents
write；不要授予 Workflows 权限。部署后分别通过使用对应 `EnvironmentFile` 的临时
systemd unit 或受控维护终端运行 `osc-agent bot doctor --control` 和
`osc-agent bot doctor --worker`。Control Doctor 不访问 Docker；Worker Doctor 校验
`git`、`docker`、`pwsh`、`rg`、模型配置和本机镜像。不要用命令替换把 env 文件展开到 argv
或 shell history；Doctor 本身不会输出凭据值。

## 服务与网络

将本目录的 unit 文件复制到 `/etc/systemd/system/`，Nginx 配置复制到站点配置后启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now osc-agent-bot-control osc-agent-bot-worker osc-agent-bot-cleanup.timer
sudo nginx -t
sudo systemctl reload nginx
```

阿里云安全组只公开 HTTPS 443；SSH 22 仅允许管理来源 IP。8080 仅监听
`127.0.0.1`。SQLite、Docker API 和任何内部端口不得公开。Control 需要访问 GitHub API，
Worker 宿主进程需要访问模型 API；Docker 仓库命令始终完全断网。

日志交给 journald，并由系统策略轮转。完成 workspace 默认 24 小时后清理，Job/审计默认
30 天后清理。备份 SQLite 时同时处理 `-wal`/`-shm`，或先停止两个服务并使用 SQLite
在线备份机制。
