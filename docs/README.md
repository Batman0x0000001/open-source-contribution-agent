# 项目文档

本文档目录以 Open Source Contribution Agent `0.3.2` 为基准，只描述当前代码、当前部署方式和当前设计约束。

## 阅读顺序

1. 从项目根目录 [`README.md`](../README.md) 了解用途、CLI 和 GitHub App Bot。
2. 阅读 [`architecture/bot-first.md`](architecture/bot-first.md) 建立生产组件全景。
3. 阅读 [`architecture/agent-runtime.md`](architecture/agent-runtime.md) 理解统一 Agent Runtime。
4. 阅读 [`architecture/tools.md`](architecture/tools.md) 理解 Tool adapter 与共享能力边界。
5. 阅读 [`architecture/current-state-schema.md`](architecture/current-state-schema.md) 理解 SQLite 和 ExecutionContract。
6. 阅读 [`operations/observability.md`](operations/observability.md) 了解健康检查、指标和故障处理。

## 架构约束

- [`architecture/architecture-invariants.md`](architecture/architecture-invariants.md)：由测试保护的核心不变量。
- [`architecture/non-goals.md`](architecture/non-goals.md)：当前明确不实现的能力。
- [`architecture/adr-auto-merge.md`](architecture/adr-auto-merge.md)：禁止自动合并 PR 的安全决策。

Ubuntu 安装、升级、权限和 Nginx 配置见 [`deploy/ubuntu/README.md`](../deploy/ubuntu/README.md)。
