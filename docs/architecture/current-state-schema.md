# 当前状态模型

本文描述 Open Source Contribution Agent `0.3.5` 的持久化状态边界。Control 与 Worker 必须使用相同的状态模型；不支持不同状态模型的进程混跑或滚动升级。

Runtime Session 使用 V6。Metadata 只保存 `session_id`、`workspace_root`、`model` 和
`system_prompt`；workspace、permissions、capabilities、completion requirements 与终态统一
保存在 `AgentRunState`。JSONL 与 SQLite Store 使用相同记录模型。V5 不兼容且不会自动迁移。
V6 system prompt 不再复制 Skill 或 Agent 发现列表；当前 Tool Schema 是可用能力的唯一模型可见
权威。恢复使用持久化 capability，当前工具注册表只能让已删除或禁用的工具不可用，不能扩大权限。

`ExecutionContract` 使用不可变的规范 JSON。它绑定仓库、GitHub App 安装、Issue、默认分支和 base SHA，以及 Issue 输入哈希、模型、Runtime、Agent profile、Skill、Tool allowlist、验证命令、拒绝路径、patch 限制、不可变 Docker image ID、容器资源和 PR 发布模式。Token、私钥、Secret、Prompt 和宿主绝对路径不属于契约字段。

`bot_jobs` 把稳定的仓库与 Issue 身份保存为索引列，并把完整类型化 Job 保存为规范 JSON。针对 `(repository_id, issue_number)` 的部分唯一索引覆盖所有活动状态，确保一个 Issue 同时只有一个活动 Job。

Plan 与 Implementation 分别记录尝试次数、Session、工作区路径和准备状态。Job 还独立保存批准、Artifact、lease、重试阶段、分支、提交、PR、错误、乐观锁版本和时间戳。进度字段只保存有界 RuntimeEvent 类型和时间；Prompt、Tool 输入输出与仓库文件内容不会写入 Job。

`bot_inbox_messages` 根据 GitHub comment ID 去重。Plan reply 会在同一个 `BEGIN IMMEDIATE` 事务中追加到 SQLite Session 事件链并标记为已消费。Plan、Approval 和 Delivery Artifact 都绑定 ExecutionContract hash 与 base SHA；Implementation Approval 还绑定 Plan evidence hash。

升级脚本采用停机归档与原子重建：先阻止新 Webhook、停止 Worker 和 Control、归档 SQLite/WAL/SHM 与工作区，再切换 release、初始化当前状态模型并执行 schema-check 和 smoke-test。归档状态只用于审计，不作为可恢复 Job 导入当前数据库。
