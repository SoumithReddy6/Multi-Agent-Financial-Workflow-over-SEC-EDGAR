"""Load and validate reusable finance workflow templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from sec_memo_agents.schemas import WorkflowTemplate


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def load_template(name: str, template_dir: Path = TEMPLATE_DIR) -> WorkflowTemplate:
    path = template_dir / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in template_dir.glob("*.yaml")))
        raise FileNotFoundError(f"Template '{name}' not found. Available templates: {available}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return WorkflowTemplate.model_validate(payload)


def list_templates(template_dir: Path = TEMPLATE_DIR) -> list[WorkflowTemplate]:
    return [load_template(path.stem, template_dir) for path in sorted(template_dir.glob("*.yaml"))]
