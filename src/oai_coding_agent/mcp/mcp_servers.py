"""
Launch and register cleanup for filesystem, CLI & Git MCP servers via AsyncExitStack.
"""

import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, List, Optional

from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


# CLI MCP server restrictions
ALLOWED_CLI_COMMANDS = [
    "grep",
    "rg",
    "find",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "pwd",
    "echo",
    "sed",
    "awk",
    "sort",
    "uniq",
    "fzf",
    "bat",
    "git",
    "uv",
    "pip",
    "pipdeptree",
    "pre-commit",
    "xargs",
    "which",
]

ALLOWED_CLI_FLAGS = ["all"]


class QuietMCPServerStdio(MCPServerStdio):
    """Variant of MCPServerStdio that silences child-process stderr."""

    def create_streams(self) -> Any:
        return stdio_client(self.params, errlog=open(os.devnull, "w"))


class QuietMCPServerSse(MCPServerSse):
    """Variant of MCPServerSse that silences child-process stderr."""

    def create_streams(self) -> Any:
        # Ensure we have a URL parameter
        if "url" not in self.params:
            raise ValueError("Missing required 'url' parameter for SSE client")
        return sse_client(self.params["url"])


async def start_mcp_servers(
    repo_path: Path,
    github_token: Optional[str],
    exit_stack: AsyncExitStack,
    mode: Optional[str] = None,
) -> List[MCPServer]:
    """
    Start filesystem, CLI, Git, and GitHub MCP servers, registering cleanup on the provided exit_stack.

    If mode is "plan", also starts the Atlassian MCP server.

    Returns a list of connected MCPServerStdio instances.
    """
    servers: List[MCPServer] = []

    # Project Context MCP server
    try:
        project_ctx_server = QuietMCPServerStdio(
            name="project-context-mcp",
            params={
                "command": "python",
                "args": [str(Path(__file__).parent / "project_context_mcp_server.py")],
                "cwd": str(repo_path),
            },
            client_session_timeout_seconds=30,
            cache_tools_list=True,
        )
        project_ctx = await exit_stack.enter_async_context(project_ctx_server)

        servers.append(project_ctx)
        logger.info("Project Context MCP server started successfully")
    except OSError:
        logger.exception("Failed to start Project Context MCP server")

    # Atlassian Official MCP server (only in plan mode)
    if mode == "plan":
        try:
            atlassian_ctx = QuietMCPServerStdio(
                name="atlassian-mcp",
                params={
                    "command": "npx",
                    "args": ["-y", "mcp-remote", "https://mcp.atlassian.com/v1/sse"],
                },
                client_session_timeout_seconds=120,
                cache_tools_list=True,
            )
            atlassian = await exit_stack.enter_async_context(atlassian_ctx)

            servers.append(atlassian)
            logger.info("Atlassian MCP server started successfully")
        except OSError:
            logger.exception("Failed to start Atlassian MCP server")

    # Filesystem MCP server
    fs_ctx = QuietMCPServerStdio(
        name="file-system-mcp",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(repo_path),
            ],
        },
        client_session_timeout_seconds=30,
        cache_tools_list=True,
    )
    fs = await exit_stack.enter_async_context(fs_ctx)

    servers.append(fs)
    logger.info("Filesystem MCP server started successfully")

    # CLI MCP server
    try:
        cli_ctx = QuietMCPServerStdio(
            name="cli-mcp-server",
            params={
                "command": "cli-mcp-server",
                "env": {
                    "ALLOWED_DIR": str(repo_path),
                    "ALLOWED_COMMANDS": ",".join(ALLOWED_CLI_COMMANDS),
                    "ALLOWED_FLAGS": ",".join(ALLOWED_CLI_FLAGS),
                    "ALLOW_SHELL_OPERATORS": "true",
                    "COMMAND_TIMEOUT": "120",
                    # set OAI_AGENT so commit-msg hook sees it
                    "OAI_AGENT": "true",
                },
            },
            client_session_timeout_seconds=120,
            cache_tools_list=True,
        )
        cli = await exit_stack.enter_async_context(cli_ctx)

        servers.append(cli)
        logger.info("CLI MCP server started successfully")
    except OSError:
        logger.exception("Failed to start CLI MCP server")

    # Git MCP server
    try:
        git_ctx = QuietMCPServerStdio(
            name="mcp-server-git",
            params={
                "command": "mcp-server-git",
                # set OAI_AGENT so commit-msg hook sees it
                "env": {"OAI_AGENT": "true"},
            },
            client_session_timeout_seconds=120,
            cache_tools_list=True,
        )
        git = await exit_stack.enter_async_context(git_ctx)

        servers.append(git)
        logger.info("Git MCP server started successfully")
    except OSError:
        logger.exception("Failed to start Git MCP server")

    # GitHub MCP server (only if token is available)
    if github_token:
        try:
            gh_ctx = QuietMCPServerStdio(
                name="github-mcp-server",
                params={
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e",
                        "GITHUB_PERSONAL_ACCESS_TOKEN",
                        "ghcr.io/github/github-mcp-server",
                    ],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
                },
                client_session_timeout_seconds=120,
                cache_tools_list=True,
            )
            gh = await exit_stack.enter_async_context(gh_ctx)

            servers.append(gh)
            logger.info("GitHub MCP server started successfully")
        except OSError:
            logger.exception("Failed to start GitHub MCP server")
    else:
        logger.info("No GitHub token available, skipping GitHub MCP server")

    return servers
