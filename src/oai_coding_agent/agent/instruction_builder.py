"""
Build dynamic instructions from templates.
"""

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from ..mcp.project_context import ProjectContext
from ..runtime_config import RuntimeConfig

TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    autoescape=False,
    keep_trailing_newline=True,
)


def build_instructions(config: RuntimeConfig, include_context: bool = False) -> str:
    """Build instructions from template based on configuration."""
    template_name = f"prompt_{config.mode.value}.jinja2"

    # Use context-enhanced template if requested
    if include_context:
        context_template = f"prompt_{config.mode.value}_with_context.jinja2"
        try:
            template = TEMPLATE_ENV.get_template(context_template)
        except TemplateNotFound:
            # Fall back to default with context
            try:
                template = TEMPLATE_ENV.get_template(
                    "prompt_default_with_context.jinja2"
                )
            except TemplateNotFound:
                # Fall back to regular template
                template = TEMPLATE_ENV.get_template(template_name)
                include_context = False
    else:
        try:
            template = TEMPLATE_ENV.get_template(template_name)
        except TemplateNotFound:
            template = TEMPLATE_ENV.get_template("prompt_default.jinja2")

    template_vars: Dict[str, Any] = {
        "repo_path": str(config.repo_path),
        "mode": config.mode.value,
        "github_repository": config.github_repo or "",
        "branch_name": config.branch_name or "",
    }

    # Add project context if requested
    if include_context:
        ctx = ProjectContext(config.repo_path)
        template_vars["project_structure"] = ctx.get_structure_summary(max_depth=2)
        template_vars["semantic_map"] = ctx.get_semantic_map()
        template_vars["entry_points"] = _find_entry_points(ctx)

    return template.render(**template_vars)


def _find_entry_points(ctx: ProjectContext) -> list[str]:
    """Find likely entry points in the project."""
    semantic_map = ctx.get_semantic_map()
    entry_points = []

    # Look for main/cli files
    for tag in ["MAIN", "CLI"]:
        if tag in semantic_map:
            entry_points.extend(semantic_map[tag][:3])

    return entry_points
