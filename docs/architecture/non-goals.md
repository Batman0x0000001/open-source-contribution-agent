# 当前架构非目标

以下能力不属于 Open Source Contribution Agent `0.3.5` 的最小核心。当前实现不为其提前建立专用框架。

| 排除项 | 原因 | 未来触发条件 |
|---|---|---|
| Ink/React 终端 UI | 当前 CLI 不需要组件渲染系统 | 出现独立交互式 TUI 需求 |
| Tool UI 渲染协议 | 与 Runtime 执行无关 | 多前端需要统一 Tool 视觉协议 |
| 遥测与实验 Feature Gate | 产品运营复杂度，不是 Agent 核心 | 出现真实灰度发布和遥测平台 |
| Tool Search/deferred tool | 当前工具规模不需要延迟发现 | Tool schema 明显挤占上下文 |
| 远程 Skill 搜索与热更新 | 当前只需本地 Skill | 出现远程 Skill 仓库与更新需求 |
| Voice、IDE、Browser、Computer Use | 不属于开源贡献核心 | 对应适配器进入明确产品范围 |
| 完整 swarm/coordinator | 当前只需受控 subagent | 单机 subagent 无法满足任务并发 |
| 企业 managed settings | 当前无企业策略分发 | 出现组织级策略管理需求 |
| 多 Provider 实现 | 当前固定 Anthropic SDK | 出现第二个真实 Provider 使用者 |
| 插件市场 | 当前本地 Tool/Skill 发现已足够 | 需要独立分发、安装和版本治理 |
| 多套实验性 context collapse | 最小 compact 管线足够 | 有数据证明现有管线无法满足窗口管理 |
| 通用 DAG/Workflow DSL | Contribution 由 Skill + Agent Loop 动态推进 | 出现多个必须确定性编排且无法由现有机制表达的消费者 |
| Task/Todo 子系统 | 当前没有 owner/dependency 的真实消费者 | 出现多 Agent 任务依赖、所有权和持久化需求 |
| Mock MCP | Mock server 不应成为生产能力 | 出现真实 Transport、生命周期和权限需求 |

排除不等于永久禁止。未来只有在出现真实使用者、验收标准和维护责任时，才在现有稳定接口上增加对应能力。
