import os
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest
from agents.mcp import MCPServerStdioParams

import oai_coding_agent.mcp.mcp_servers as mcp_servers


class DummyExitStack:
    """Dummy exit stack to capture pushed async callbacks."""

    callbacks: list[tuple[Callable[..., Any], tuple[Any, ...]]]

    def __init__(self) -> None:
        self.callbacks = []

    def push_async_callback(self, func: Callable[..., Any], *args: Any) -> None:
        self.callbacks.append((func, args))

    async def enter_async_context(self, ctx: Any) -> Any:
        """Enter the async context and register exit callback."""
        result = await ctx.__aenter__()
        self.push_async_callback(ctx.__aexit__, None, None, None)
        return result


def test_create_streams_uses_stdio_client_and_devnull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    QuietMCPServerStdio.create_streams should call stdio_client with the instance params
    and an errlog pointing to os.devnull in write mode.
    """
    calls: list[tuple[Any, Any]] = []

    def fake_stdio_client(params: MCPServerStdioParams, errlog: Any) -> str:
        calls.append((params, errlog))
        return "STREAMS"

    monkeypatch.setattr(mcp_servers, "stdio_client", fake_stdio_client)

    params: dict[str, str] = {"command": "dummy"}
    ctx = mcp_servers.QuietMCPServerStdio(
        name="test",
        params=cast(MCPServerStdioParams, params),
        client_session_timeout_seconds=1,
        cache_tools_list=False,
    )
    streams = ctx.create_streams()

    assert streams == "STREAMS"
    assert len(calls) == 1
    called_params, errlog = calls[0]
    assert (
        hasattr(called_params, "command") and called_params.command == params["command"]
    )
    assert hasattr(errlog, "name")
    assert errlog.name == os.devnull
    assert errlog.mode == "w"


@pytest.mark.asyncio
async def test_start_mcp_servers_all_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    All three MCP servers should be started successfully when no errors occur.
    """

    # Dummy context manager to replace QuietMCPServerStdio
    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/some/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), None
    )
    # Should start filesystem, CLI, git, and GitHub servers
    names = [s.name for s in servers]
    assert names == [
        "project-context-mcp",
        "file-system-mcp",
        "cli-mcp-server",
        "mcp-server-git",
        "github-mcp-server",
    ]
    # exit_stack should have a callback for each server
    assert len(exit_stack.callbacks) == 5


@pytest.mark.asyncio
async def test_start_mcp_servers_skip_cli_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the CLI MCP server raises OSError, it should be skipped and not added to the list.
    """
    fail_names = {"cli-mcp-server"}

    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            if self.name in fail_names:
                raise OSError("CLI failure")
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), None
    )
    names = [s.name for s in servers]
    # Should skip CLI and include filesystem, git, and GitHub servers
    assert names == [
        "project-context-mcp",
        "file-system-mcp",
        "mcp-server-git",
        "github-mcp-server",
    ]
    assert len(exit_stack.callbacks) == 4


@pytest.mark.asyncio
async def test_start_mcp_servers_skip_git_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the Git MCP server raises OSError, it should be skipped and not added to the list.
    """
    fail_names = {"mcp-server-git"}

    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            if self.name in fail_names:
                raise OSError("Git failure")
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), None
    )
    names = [s.name for s in servers]
    # Should skip Git and include filesystem, CLI, and GitHub servers
    assert names == [
        "project-context-mcp",
        "file-system-mcp",
        "cli-mcp-server",
        "github-mcp-server",
    ]
    assert len(exit_stack.callbacks) == 4


@pytest.mark.asyncio
async def test_start_mcp_servers_skip_cli_and_git_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If both CLI and Git MCP servers raise OSError, only filesystem should start.
    """
    fail_names = {"cli-mcp-server", "mcp-server-git"}

    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            if self.name in fail_names:
                raise OSError("failure")
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), None
    )
    names = [s.name for s in servers]
    # Should skip CLI and Git and include filesystem and GitHub servers only
    assert names == ["project-context-mcp", "file-system-mcp", "github-mcp-server"]
    assert len(exit_stack.callbacks) == 3


@pytest.mark.asyncio
async def test_start_mcp_servers_plan_mode_includes_atlassian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    In plan mode, the Atlassian MCP server should be started.
    """

    # Dummy context manager to replace QuietMCPServerStdio
    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/some/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), "plan"
    )
    # Should start Atlassian, filesystem, CLI, git, and GitHub servers
    names = [s.name for s in servers]
    assert names == [
        "project-context-mcp",
        "atlassian-mcp",
        "file-system-mcp",
        "cli-mcp-server",
        "mcp-server-git",
        "github-mcp-server",
    ]
    # exit_stack should have a callback for each server
    assert len(exit_stack.callbacks) == 6


@pytest.mark.asyncio
async def test_start_mcp_servers_plan_mode_skip_atlassian_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    In plan mode, if Atlassian MCP server fails, it should be skipped.
    """
    fail_names = {"atlassian-mcp"}

    class DummyCtx:
        def __init__(
            self,
            name: str,
            params: Any,
            client_session_timeout_seconds: int | None = None,
            cache_tools_list: bool | None = None,
        ) -> None:
            self.name = name
            self.params = params

        async def __aenter__(self) -> SimpleNamespace:
            if self.name in fail_names:
                raise OSError("Atlassian failure")
            return SimpleNamespace(name=self.name, params=self.params)

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    monkeypatch.setattr(mcp_servers, "QuietMCPServerStdio", DummyCtx)

    exit_stack = DummyExitStack()
    repo = Path("/repo")

    servers = await mcp_servers.start_mcp_servers(
        repo, "dummy-token", cast(AsyncExitStack[bool | None], exit_stack), "plan"
    )
    names = [s.name for s in servers]
    # Should skip Atlassian and include filesystem, CLI, git, and GitHub servers
    assert names == [
        "project-context-mcp",
        "file-system-mcp",
        "cli-mcp-server",
        "mcp-server-git",
        "github-mcp-server",
    ]
    assert len(exit_stack.callbacks) == 5
