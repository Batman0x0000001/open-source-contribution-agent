"""统一读取独立配置与生产组合配置的 YAML section。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_PRODUCTION_SECTIONS = frozenset({"runtime", "repositories"})


def load_config_section(
    path: Path,
    *,
    section: str,
    source_name: str,
) -> Any:
    """读取 YAML，并在组合配置中选择唯一的业务 section。"""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to read {source_name} config {path}: {exc}") from exc
    if not isinstance(raw, dict) or not (_PRODUCTION_SECTIONS & set(raw)):
        return raw
    unknown = set(raw) - _PRODUCTION_SECTIONS
    if unknown:
        raise ValueError(
            f"unknown production config sections: {', '.join(sorted(unknown))}"
        )
    if section not in raw:
        raise ValueError(f"production config is missing {section} section")
    return raw[section]
