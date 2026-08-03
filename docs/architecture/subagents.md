# 子 Agent 能力层

`osc_agent/subagents/` 只负责把有界任务作为子 Session 交给同一个
`AgentRuntime.query()` 执行。它不是 CLI/Bot 入口，也不拥有独立模型循环。

## 阅读顺序

1. `models.py`：静态定义、单次请求和运行结果。
2. `registry.py`：代码内置子 Agent 的严格注册契约。
3. `runner.py`：上下文策略、capability 收窄和子 Session 生命周期。
4. `tool.py`：模型可见的 `agent` Tool 协议、限流和结构化输出校验。
5. `builtins/`：Explore 与 Verify 的具体输入、输出和 Prompt。

## 数据流

```text
AgentRuntime.query(parent)
  → AgentTool.call()
  → SubagentRunner.run()
  → AgentRuntime.query(child session)
  → AgentToolResult(child_session_id, typed result)
```

Application 是组装边界：它创建唯一 Runtime、Registry 和 Runner，再把 `AgentTool`
注册到 Runtime。Runtime 不反向依赖 `subagents`。

## 不变量

- 每个子运行使用独立 Session，但复用同一个 Runtime 对象和工具执行链。
- `minimal` 不复制父 transcript；`fork` 必须有父消息并使用深拷贝。
- 最终 capability 是调用者与定义的交集，并无条件移除 `agent`。
- `read_only` 是声明式只读加 Git 可见变更检测，不是操作系统级写入隔离。
- 父 Session 通过 ToolResult 中的 `child_session_id` 持有子运行关系。
