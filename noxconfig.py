from __future__ import annotations

from pathlib import Path

from exasol.toolbox.config import BaseConfig

PROJECT_CONFIG = BaseConfig(
    root_path=Path(__file__).parent,
    project_name="ai/text",
    python_versions=("3.11", "3.12"),
    add_to_excluded_python_paths=("_experiments",),
)
