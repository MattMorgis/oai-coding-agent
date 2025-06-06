"""
Project context analyzer for providing rich codebase information to AI agents.
"""

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORE_FILES = {".DS_Store", ".gitignore", "*.pyc", "*.pyo", "*.egg-info"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".go",
    ".rs",
}


@dataclass
class FileInfo:
    path: Path
    size: int
    lines: int
    imports: Set[str] = field(default_factory=set)
    exports: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)


@dataclass
class DirInfo:
    path: Path
    file_count: int
    total_lines: int
    tags: List[str] = field(default_factory=list)
    children: Dict[str, "DirInfo"] = field(default_factory=dict)
    files: List[FileInfo] = field(default_factory=list)


class ProjectContext:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self._file_cache: Dict[Path, FileInfo] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._semantic_tags: Dict[str, List[Path]] = defaultdict(list)

    def get_structure_summary(
        self, path: Optional[str] = None, max_depth: int = 5
    ) -> str:
        """Generate a rich directory tree with metadata."""
        target_path = Path(path) if path else self.root_path
        if not target_path.is_absolute():
            target_path = self.root_path / target_path

        dir_info = self._analyze_directory(
            target_path, current_depth=0, max_depth=max_depth
        )
        return self._format_structure(dir_info)

    def get_dependency_graph(self, entry_point: str) -> Dict[str, List[str]]:
        """Generate import dependency graph starting from an entry point."""
        entry_path = self._resolve_path(entry_point)
        if not entry_path or not entry_path.exists():
            return {}

        visited: Set[Path] = set()
        graph: Dict[str, List[str]] = defaultdict(list)
        self._trace_dependencies(entry_path, graph, visited)
        return dict(graph)

    def get_semantic_map(self) -> Dict[str, List[str]]:
        """Group files by functionality based on naming and content patterns."""
        self._scan_project_semantics()
        return {
            tag: [str(p.relative_to(self.root_path)) for p in paths]
            for tag, paths in self._semantic_tags.items()
        }

    def get_file_metrics(self, path: str) -> Dict[str, Any]:
        """Get metrics for a specific file."""
        file_path = self._resolve_path(path)
        if not file_path or not file_path.exists():
            return {}

        info = self._analyze_file(file_path)
        return {
            "path": str(file_path.relative_to(self.root_path)),
            "lines": info.lines,
            "size_kb": info.size / 1024,
            "imports": list(info.imports),
            "exports": list(info.exports),
            "tags": info.tags,
            "complexity": self._estimate_complexity(file_path),
        }

    def get_relevant_context(self, task_description: str) -> Dict[str, Any]:
        """Get context relevant to a specific task."""
        keywords = self._extract_keywords(task_description)
        relevant_files = self._find_relevant_files(keywords)

        context: Dict[str, Any] = {
            "keywords": keywords,
            "relevant_files": [],
            "suggested_entry_points": [],
            "related_modules": [],
        }

        for file_path in relevant_files[:10]:  # Top 10 most relevant
            info = self._analyze_file(file_path)
            rel_path = str(file_path.relative_to(self.root_path))
            context["relevant_files"].append(
                {
                    "path": rel_path,
                    "tags": info.tags,
                    "imports": list(info.imports)[:5],
                }
            )

            if "MAIN" in info.tags or "CLI" in info.tags:
                context["suggested_entry_points"].append(rel_path)

        # Find related modules
        for file_path in relevant_files[:5]:
            deps = self.get_dependency_graph(str(file_path))
            for dep in deps.get(str(file_path), [])[:3]:
                if dep not in context["related_modules"]:
                    context["related_modules"].append(dep)

        return context

    def _analyze_directory(
        self, path: Path, current_depth: int, max_depth: int
    ) -> DirInfo:
        """Recursively analyze directory structure."""
        dir_info = DirInfo(path=path, file_count=0, total_lines=0)

        if current_depth >= max_depth:
            return dir_info

        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith(".") or item.name in IGNORE_DIRS:
                    continue

                if item.is_dir():
                    if item.name not in IGNORE_DIRS:
                        child_info = self._analyze_directory(
                            item, current_depth + 1, max_depth
                        )
                        dir_info.children[item.name] = child_info
                        dir_info.file_count += child_info.file_count
                        dir_info.total_lines += child_info.total_lines
                elif item.is_file() and item.suffix in CODE_EXTENSIONS:
                    file_info = self._analyze_file(item)
                    dir_info.files.append(file_info)
                    dir_info.file_count += 1
                    dir_info.total_lines += file_info.lines

        except PermissionError:
            pass

        # Add semantic tags to directory
        dir_info.tags = self._infer_dir_tags(path, dir_info)

        return dir_info

    def _analyze_file(self, path: Path) -> FileInfo:
        """Analyze a single file."""
        if path in self._file_cache:
            return self._file_cache[path]

        try:
            size = path.stat().st_size
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = len(content.splitlines())

            info = FileInfo(path=path, size=size, lines=lines)

            if path.suffix == ".py":
                self._extract_python_info(content, info)

            info.tags = self._infer_file_tags(path, info)
            self._file_cache[path] = info

            return info
        except Exception:
            return FileInfo(path=path, size=0, lines=0)

    def _extract_python_info(self, content: str, info: FileInfo) -> None:
        """Extract imports and exports from Python files."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info.imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        info.imports.add(node.module.split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    info.exports.add(node.name)
        except Exception:
            pass

    def _infer_file_tags(self, path: Path, info: FileInfo) -> List[str]:
        """Infer semantic tags from file name and content."""
        tags = []
        name_lower = path.stem.lower()

        # Name-based tags
        if "test" in name_lower:
            tags.append("TEST")
        if "config" in name_lower:
            tags.append("CONFIG")
        if name_lower in ["main", "__main__", "cli", "app"]:
            tags.append("MAIN")
        if "auth" in name_lower:
            tags.append("AUTH")
        if "api" in name_lower or "route" in name_lower:
            tags.append("API")
        if "model" in name_lower or "schema" in name_lower:
            tags.append("MODEL")
        if "util" in name_lower or "helper" in name_lower:
            tags.append("UTIL")

        # Import-based tags
        if "asyncio" in info.imports or "async" in str(path):
            tags.append("ASYNC")
        if "requests" in info.imports or "httpx" in info.imports:
            tags.append("HTTP")

        return tags

    def _infer_dir_tags(self, path: Path, dir_info: DirInfo) -> List[str]:
        """Infer semantic tags for directories."""
        tags = []
        name_lower = path.name.lower()

        if "test" in name_lower:
            tags.append("TEST")
        elif "doc" in name_lower:
            tags.append("DOC")
        elif name_lower in ["src", "lib", "core"]:
            tags.append("CORE")
        elif name_lower in ["api", "routes", "endpoints"]:
            tags.append("API")
        elif name_lower in ["models", "schemas", "entities"]:
            tags.append("MODEL")

        return tags

    def _format_structure(
        self, dir_info: DirInfo, prefix: str = "", is_last: bool = True
    ) -> str:
        """Format directory structure as a tree."""
        lines = []

        # Current directory
        name = dir_info.path.name
        if prefix == "":
            line = f"{name}/"
        else:
            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{name}/"

        # Add metadata
        if dir_info.tags:
            line += f" [{', '.join(dir_info.tags)}]"
        if dir_info.file_count > 0:
            line += f" ({dir_info.file_count} files, ~{dir_info.total_lines} LOC)"

        lines.append(line)

        # Add children
        if prefix == "":
            child_prefix = ""
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Process subdirectories
        children = list(dir_info.children.items())
        for i, (name, child) in enumerate(children):
            is_last_child = i == len(children) - 1 and len(dir_info.files) == 0
            lines.append(self._format_structure(child, child_prefix, is_last_child))

        # Process files (show only first few)
        files_to_show = dir_info.files[:3]
        for i, file_info in enumerate(files_to_show):
            is_last_file = i == len(files_to_show) - 1
            connector = "└── " if is_last_file else "├── "
            file_line = f"{child_prefix}{connector}{file_info.path.name}"

            if file_info.tags:
                file_line += f" [{', '.join(file_info.tags)}]"

            # Add import indicators for key files
            if file_info.imports:
                key_imports = list(file_info.imports)[:2]
                if key_imports:
                    file_line += f" → {{{', '.join(key_imports)}}}"

            lines.append(file_line)

        if len(dir_info.files) > 3:
            lines.append(
                f"{child_prefix}└── ... and {len(dir_info.files) - 3} more files"
            )

        return "\n".join(lines)

    def _resolve_path(self, path_str: str) -> Optional[Path]:
        """Resolve a path string to an absolute Path object."""
        path = Path(path_str)
        if path.is_absolute():
            return path if path.exists() else None
        else:
            full_path = self.root_path / path
            return full_path if full_path.exists() else None

    def _trace_dependencies(
        self, file_path: Path, graph: Dict[str, List[str]], visited: Set[Path]
    ) -> None:
        """Trace import dependencies recursively."""
        if file_path in visited:
            return

        visited.add(file_path)
        file_info = self._analyze_file(file_path)
        rel_path = str(file_path.relative_to(self.root_path))

        for imp in file_info.imports:
            # Try to resolve local imports
            possible_paths = [
                file_path.parent / f"{imp}.py",
                file_path.parent / imp / "__init__.py",
                self.root_path / f"{imp}.py",
                self.root_path / imp / "__init__.py",
            ]

            for possible_path in possible_paths:
                if possible_path.exists() and possible_path != file_path:
                    dep_rel_path = str(possible_path.relative_to(self.root_path))
                    graph[rel_path].append(dep_rel_path)
                    self._trace_dependencies(possible_path, graph, visited)
                    break

    def _scan_project_semantics(self) -> None:
        """Scan project and build semantic groupings."""
        for root, dirs, files in os.walk(self.root_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                if file_path.suffix in CODE_EXTENSIONS:
                    info = self._analyze_file(file_path)
                    for tag in info.tags:
                        self._semantic_tags[tag].append(file_path)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from task description."""
        # Simple keyword extraction - in production, use NLP
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in common_words]
        return keywords

    def _find_relevant_files(self, keywords: List[str]) -> List[Path]:
        """Find files relevant to given keywords."""
        scores: Dict[Path, int] = defaultdict(int)

        for file_path, info in self._file_cache.items():
            rel_path = str(file_path.relative_to(self.root_path)).lower()

            # Score based on path matches
            for keyword in keywords:
                if keyword in rel_path:
                    scores[file_path] += 3

            # Score based on tag matches
            for tag in info.tags:
                for keyword in keywords:
                    if keyword.upper() in tag:
                        scores[file_path] += 2

        # Sort by score
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

    def _estimate_complexity(self, file_path: Path) -> int:
        """Estimate cyclomatic complexity (simplified)."""
        try:
            content = file_path.read_text()
            # Very simplified - count decision points
            complexity = 1
            complexity += content.count("if ")
            complexity += content.count("elif ")
            complexity += content.count("for ")
            complexity += content.count("while ")
            complexity += content.count("except ")
            return complexity
        except Exception:
            return 0
