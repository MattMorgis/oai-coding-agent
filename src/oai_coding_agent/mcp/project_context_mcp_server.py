#!/usr/bin/env python3
"""
Standalone MCP server for project context information.
This runs as a separate process and communicates via stdio.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import mcp.server.stdio
from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

from oai_coding_agent.mcp.project_context import ProjectContext

server: Any = Server("project-context")

# Global dictionary of dynamic context managers keyed by repo path
dynamic_context_managers: Dict[str, Any] = {}


@server.list_tools()  # type: ignore[misc]
async def list_tools() -> List[Tool]:
    """List available project context tools."""
    return [
        Tool(
            name="get_project_structure",
            description="Get a rich directory tree with metadata for the project or a specific path",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root (optional)",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (default: 5)",
                        "default": 5,
                    },
                },
            },
        ),
        Tool(
            name="get_dependency_graph",
            description="Get import dependency graph starting from an entry point",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_point": {
                        "type": "string",
                        "description": "Path to the entry point file",
                    }
                },
                "required": ["entry_point"],
            },
        ),
        Tool(
            name="get_semantic_map",
            description="Get files grouped by functionality (AUTH, API, TEST, etc.)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_file_metrics",
            description="Get detailed metrics for a specific file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="get_relevant_context",
            description="Get context relevant to a specific task or feature",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task or feature",
                    }
                },
                "required": ["task_description"],
            },
        ),
        Tool(
            name="start_task",
            description="Start tracking a new task for dynamic context",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the task being started",
                    }
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="record_file_access",
            description="Record that a file was accessed during the current task",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file that was accessed",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Type of operation (read, edit, create)",
                        "default": "read",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="get_dynamic_context",
            description="Get current dynamic context including recent activity and suggestions",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_session_summary",
            description="Get a summary of the current session's activity",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()  # type: ignore[misc]
async def call_tool(
    name: str, arguments: Dict[str, Any] | None = None
) -> List[TextContent]:
    """Execute a tool and return results."""
    arguments = arguments or {}

    # Get repo path from environment or current directory
    repo_path = Path.cwd()
    context = ProjectContext(repo_path)

    # Get or create dynamic context manager for this repo
    repo_path_str = str(repo_path)
    if repo_path_str not in dynamic_context_managers:
        # Import here to avoid top-level circular dependency and allow running as script
        from oai_coding_agent.mcp.dynamic_context import (  # noqa: PLC0415
            DynamicContextManager,
        )

        dynamic_context_managers[repo_path_str] = DynamicContextManager(repo_path)
    manager = dynamic_context_managers[repo_path_str]

    try:
        if name == "get_project_structure":
            result = context.get_structure_summary(
                path=arguments.get("path"), max_depth=arguments.get("max_depth", 5)
            )
            return [TextContent(type="text", text=result)]

        elif name == "get_dependency_graph":
            dep_result = context.get_dependency_graph(arguments["entry_point"])
            formatted = _format_dependency_graph(dep_result)
            return [TextContent(type="text", text=formatted)]

        elif name == "get_semantic_map":
            sem_result = context.get_semantic_map()
            formatted = _format_semantic_map(sem_result)
            return [TextContent(type="text", text=formatted)]

        elif name == "get_file_metrics":
            metrics_result = context.get_file_metrics(arguments["path"])
            formatted = json.dumps(metrics_result, indent=2)
            return [TextContent(type="text", text=formatted)]

        elif name == "get_relevant_context":
            ctx_result = context.get_relevant_context(arguments["task_description"])
            formatted = _format_relevant_context(ctx_result)
            return [TextContent(type="text", text=formatted)]

        elif name == "start_task":
            manager.start_task(arguments["description"])
            return [
                TextContent(
                    type="text",
                    text=f"Started tracking task: {arguments['description']}",
                )
            ]

        elif name == "record_file_access":
            operation = arguments.get("operation", "read")
            manager.record_file_access(arguments["file_path"], operation)
            return [
                TextContent(
                    type="text",
                    text=f"Recorded {operation} access to {arguments['file_path']}",
                )
            ]

        elif name == "get_dynamic_context":
            dynamic_ctx = manager.get_current_context()
            formatted = _format_dynamic_context(dynamic_ctx)
            return [TextContent(type="text", text=formatted)]

        elif name == "get_session_summary":
            summary = manager.export_session_summary()
            formatted = json.dumps(summary, indent=2)
            return [TextContent(type="text", text=formatted)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def _format_dependency_graph(graph: Dict[str, List[str]]) -> str:
    """Format dependency graph for display."""
    lines = ["Dependency Graph:"]
    for file, deps in graph.items():
        if deps:
            lines.append(f"\n{file}:")
            for dep in deps:
                lines.append(f"  → {dep}")
    return "\n".join(lines)


def _format_semantic_map(semantic_map: Dict[str, List[str]]) -> str:
    """Format semantic map for display."""
    lines = ["Semantic Map:"]
    for tag, files in semantic_map.items():
        if files:
            lines.append(f"\n[{tag}] files:")
            for f in files[:5]:  # Show first 5
                lines.append(f"  - {f}")
            if len(files) > 5:
                lines.append(f"  ... and {len(files) - 5} more")
    return "\n".join(lines)


def _format_relevant_context(context: Dict[str, Any]) -> str:
    """Format relevant context for display."""
    lines = [f"Task: {' '.join(context['keywords'])}"]

    if context["relevant_files"]:
        lines.append("\nRelevant files:")
        for f in context["relevant_files"]:
            tags = f"{f['tags']}" if f["tags"] else ""
            lines.append(f"  - {f['path']} {tags}")

    if context["suggested_entry_points"]:
        lines.append("\nSuggested entry points:")
        for ep in context["suggested_entry_points"]:
            lines.append(f"  - {ep}")

    if context["related_modules"]:
        lines.append("\nRelated modules:")
        for mod in context["related_modules"]:
            lines.append(f"  - {mod}")

    return "\n".join(lines)


def _format_dynamic_context(context: Dict[str, Any]) -> str:
    """Format dynamic context for display."""
    lines = ["Dynamic Context:"]

    if context["current_task"]:
        task = context["current_task"]
        lines.append(f"\nCurrent Task: {task['description']}")
        if task["files_accessed"]:
            lines.append("Files accessed:")
            for f in task["files_accessed"]:
                lines.append(f"  - {f}")
        if task["files_modified"]:
            lines.append("Files modified:")
            for f in task["files_modified"]:
                lines.append(f"  - {f}")

    if context["recently_accessed"]:
        lines.append("\nRecently accessed files:")
        for f in context["recently_accessed"]:
            lines.append(f"  - {f}")

    if context["frequently_accessed"]:
        lines.append("\nFrequently accessed files:")
        for f in context["frequently_accessed"]:
            lines.append(f"  - {f['path']} (accessed {f['count']} times)")

    if context["related_files"]:
        lines.append("\nRelated files (based on dependencies):")
        for f in context["related_files"]:
            lines.append(f"  - {f}")

    if context["suggested_next_files"]:
        lines.append("\nSuggested next files:")
        for f in context["suggested_next_files"]:
            lines.append(f"  - {f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    asyncio.run(main())
