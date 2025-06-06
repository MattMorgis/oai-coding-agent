"""
Dynamic context manager that updates based on agent interactions.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..agent.events import (
    MessageOutputEvent,
    ReasoningEvent,
    ToolCallEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class FileAccess:
    """Track file access patterns."""

    path: Path
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    operations: List[str] = field(default_factory=list)  # read, edit, create


@dataclass
class TaskContext:
    """Context for a specific task."""

    description: str
    files_accessed: Set[Path] = field(default_factory=set)
    files_modified: Set[Path] = field(default_factory=set)
    relevant_modules: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)


class DynamicContextManager:
    """Manages dynamic context that evolves during agent interactions."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        # Import ProjectContext here to avoid circular dependency
        from oai_coding_agent.mcp.project_context import ProjectContext  # noqa: PLC0415

        self.project_context = ProjectContext(repo_path)

        # Track access patterns
        self.file_access: Dict[Path, FileAccess] = {}
        self.current_task: Optional[TaskContext] = None
        self.task_history: List[TaskContext] = []

        # Track relationships discovered during session
        self.discovered_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.discovered_patterns: Dict[str, List[str]] = defaultdict(list)

    def start_task(self, description: str) -> None:
        """Start tracking a new task."""
        if self.current_task:
            self.task_history.append(self.current_task)
        self.current_task = TaskContext(description=description)

    def record_file_access(self, file_path: str, operation: str = "read") -> None:
        """Record that a file was accessed."""
        path = self._resolve_path(file_path)
        if not path:
            return

        # Update access tracking
        if path not in self.file_access:
            self.file_access[path] = FileAccess(path=path)

        access = self.file_access[path]
        access.access_count += 1
        access.last_accessed = datetime.now()
        access.operations.append(operation)

        # Update current task
        if self.current_task:
            self.current_task.files_accessed.add(path)
            if operation in ["edit", "create"]:
                self.current_task.files_modified.add(path)

    def record_dependency(self, from_file: str, to_file: str) -> None:
        """Record a discovered dependency relationship."""
        from_path = self._resolve_path(from_file)
        to_path = self._resolve_path(to_file)

        if from_path and to_path:
            rel_from = str(from_path.relative_to(self.repo_path))
            rel_to = str(to_path.relative_to(self.repo_path))
            self.discovered_dependencies[rel_from].add(rel_to)

    def get_current_context(self) -> Dict[str, Any]:
        """Get context relevant to current work."""
        context: Dict[str, Any] = {
            "current_task": None,
            "recently_accessed": [],
            "frequently_accessed": [],
            "related_files": [],
            "suggested_next_files": [],
        }

        # Current task info
        if self.current_task:
            context["current_task"] = {
                "description": self.current_task.description,
                "files_accessed": [
                    str(p.relative_to(self.repo_path))
                    for p in list(self.current_task.files_accessed)[:10]
                ],
                "files_modified": [
                    str(p.relative_to(self.repo_path))
                    for p in self.current_task.files_modified
                ],
            }

        # Recently accessed files
        recent = sorted(
            [fa for fa in self.file_access.values() if fa.last_accessed],
            key=lambda x: x.last_accessed or datetime.min,
            reverse=True,
        )[:5]
        context["recently_accessed"] = [
            str(fa.path.relative_to(self.repo_path)) for fa in recent
        ]

        # Frequently accessed files
        frequent = sorted(
            self.file_access.values(), key=lambda x: x.access_count, reverse=True
        )[:5]
        context["frequently_accessed"] = [
            {"path": str(fa.path.relative_to(self.repo_path)), "count": fa.access_count}
            for fa in frequent
        ]

        # Related files based on current work
        if self.current_task and self.current_task.files_accessed:
            # Get dependencies of accessed files
            related = set()
            for file_path in list(self.current_task.files_accessed)[:3]:
                rel_path = str(file_path.relative_to(self.repo_path))
                deps = self.project_context.get_dependency_graph(rel_path)
                for dep_list in deps.values():
                    related.update(dep_list[:3])

            context["related_files"] = list(related)[:10]

        # Suggest next files based on patterns
        context["suggested_next_files"] = self._suggest_next_files()

        return context

    def get_enhanced_instructions(self) -> str:
        """Get instructions enhanced with dynamic context."""
        context = self.get_current_context()

        lines = []

        if context["current_task"]:
            lines.append(f"## Current Task: {context['current_task']['description']}")
            if context["current_task"]["files_modified"]:
                lines.append("\nFiles modified so far:")
                for f in context["current_task"]["files_modified"]:
                    lines.append(f"  - {f}")

        if context["recently_accessed"]:
            lines.append("\n## Recently Accessed Files")
            for f in context["recently_accessed"]:
                lines.append(f"  - {f}")

        if context["related_files"]:
            lines.append("\n## Related Files (based on dependencies)")
            for f in context["related_files"][:5]:
                lines.append(f"  - {f}")

        if context["suggested_next_files"]:
            lines.append("\n## Suggested Files to Review")
            for f in context["suggested_next_files"]:
                lines.append(f"  - {f}")

        return "\n".join(lines)

    def _suggest_next_files(self) -> List[str]:
        """Suggest files that might be relevant next."""
        suggestions: List[str] = []

        if not self.current_task:
            return suggestions

        # Look for test files if we modified source files
        for modified in self.current_task.files_modified:
            if "test" not in str(modified):
                # Suggest corresponding test file
                test_path = self._find_test_file(modified)
                if test_path:
                    suggestions.append(str(test_path.relative_to(self.repo_path)))

        # Look for related config files
        if any("config" in str(f) for f in self.current_task.files_accessed):
            # Already accessing config
            pass
        else:
            # Suggest config files
            config_files = self.project_context.get_semantic_map().get("CONFIG", [])
            suggestions.extend(config_files[:2])

        return suggestions[:5]

    def _find_test_file(self, source_file: Path) -> Optional[Path]:
        """Find test file for a source file."""
        # Common test file patterns
        patterns = [
            source_file.parent / f"test_{source_file.name}",
            source_file.parent / "tests" / f"test_{source_file.name}",
            self.repo_path
            / "tests"
            / source_file.relative_to(self.repo_path).parent
            / f"test_{source_file.name}",
        ]

        for pattern in patterns:
            if pattern.exists():
                return pattern

        return None

    def _resolve_path(self, path_str: str) -> Optional[Path]:
        """Resolve a path string to an absolute Path object."""
        path = Path(path_str)
        if path.is_absolute():
            return path if path.exists() else None
        else:
            full_path = self.repo_path / path
            return full_path if full_path.exists() else None

    def export_session_summary(self) -> Dict[str, Any]:
        """Export a summary of the session for future reference."""
        return {
            "tasks_completed": len(self.task_history),
            "current_task": self.current_task.description
            if self.current_task
            else None,
            "files_accessed_count": len(self.file_access),
            "files_modified": [
                str(p.relative_to(self.repo_path))
                for p in self.file_access.keys()
                if any(
                    "edit" in op or "create" in op
                    for op in self.file_access[p].operations
                )
            ],
            "discovered_dependencies": {
                k: list(v) for k, v in self.discovered_dependencies.items()
            },
            "access_patterns": {
                str(p.relative_to(self.repo_path)): {
                    "count": fa.access_count,
                    "operations": list(set(fa.operations)),
                }
                for p, fa in self.file_access.items()
            },
        }

    def track_tool_call(
        self, event: ToolCallEvent | ReasoningEvent | MessageOutputEvent
    ) -> None:
        """Track tool calls for dynamic context awareness."""
        logger.info(f"Tracking tool call: {event}")
        logger.info(f"Current task: {self.current_task}")
        if not self.current_task or not isinstance(event, ToolCallEvent):
            return

        try:
            args = json.loads(event.arguments) if event.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        # Track file operations based on tool name and arguments
        if event.name in ["fs_read_file", "read_file"]:
            if "path" in args:
                self.record_file_access(args["path"], "read")
        elif event.name in ["fs_write_file", "write_file"]:
            if "path" in args:
                self.record_file_access(args["path"], "create")
        elif event.name in ["fs_edit_file", "edit_file"]:
            if "path" in args:
                self.record_file_access(args["path"], "edit")
        elif event.name in ["fs_list_directory", "list_directory"]:
            if "path" in args:
                self.record_file_access(args["path"], "read")
        elif event.name == "shell":
            # Try to extract file operations from shell commands
            self._track_shell_command(event.arguments)

    def _track_shell_command(self, command: str) -> None:
        """Track file operations in shell commands."""
        if not self.current_task:
            return

        # Simple heuristics for common file operations
        # This could be enhanced with more sophisticated parsing
        words = command.split()

        for i, word in enumerate(words):
            # Look for file operations
            if word in ["cat", "less", "head", "tail", "grep"]:
                # These are read operations
                for j in range(i + 1, len(words)):
                    if not words[j].startswith("-") and "/" in words[j]:
                        self.record_file_access(words[j], "read")
                        break
            elif word in ["vim", "nano", "emacs", "code"]:
                # These are edit operations
                for j in range(i + 1, len(words)):
                    if not words[j].startswith("-") and "/" in words[j]:
                        self.record_file_access(words[j], "edit")
                        break
            elif word in ["touch", "mkdir"]:
                # These are create operations
                for j in range(i + 1, len(words)):
                    if not words[j].startswith("-") and "/" in words[j]:
                        self.record_file_access(words[j], "create")
                        break
