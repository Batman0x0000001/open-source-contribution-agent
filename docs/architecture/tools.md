# Tool 与共享执行能力

`osc_agent/tools/` 只保存模型可调用 Tool 的输入输出契约和薄适配逻辑。文件系统、Git 状态、路径策略与进程执行属于共享能力，分别位于 `workspaces/` 和 `processes/`。

## 阅读顺序

1. `osc_agent/runtime/tool.py`：Tool 协议、默认行为和注册表。
2. `osc_agent/runtime/tool_execution.py`：统一的校验、权限、Hook 和调用流水线。
3. `osc_agent/tools/registry.py`：核心 Tool 的显式组装入口。
4. `osc_agent/tools/<domain>.py`：模型可见 schema、权限属性和结果映射。
5. `osc_agent/workspaces/`、`osc_agent/processes/`：可被 Runtime、Bot 和 Subagent 复用的底层能力。

数据流如下：

```text
Model ToolUse
    → ToolExecutor
    → Tool adapter
    → Workspace / Process capability
    → ToolResult
```

## 依赖边界

- Tool adapter 可以依赖 Runtime Tool 接口以及 Workspace/Process 能力。
- Runtime、Bot、Subagent、Workspace 和 Process 不得反向依赖具体 Tool 实现。
- Application 显式构建核心 Tool，再注册 Skill、Subagent 和产品扩展 Tool。
- `AskUserQuestionTool` 持有产品注入的 question handler；`ToolExecutor` 不按工具名分支。

该结构借鉴 Claude Code 的 Tool 自包含、工具池显式组装和共享执行能力外置原则。当前项目不复制其 React 渲染职责、巨型 `ToolUseContext`、每工具目录或动态 MCP Tool pool。

