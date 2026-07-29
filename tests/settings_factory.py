"""测试专用的已加载 AgentSettings 工厂。"""

from pathlib import Path
from typing import Any

from osc_agent.configuration import (
    AgentSettings,
    RuntimeConfig,
    default_runtime_config_path,
    load_runtime_config,
)


def make_agent_settings(**overrides: Any) -> AgentSettings:
    path = Path(overrides.pop("runtime_config_path", default_runtime_config_path()))
    runtime = overrides.pop("runtime", None)
    if runtime is None:
        runtime = load_runtime_config(path)
    assert isinstance(runtime, RuntimeConfig)
    return AgentSettings(runtime_config_path=path, runtime=runtime, **overrides)
