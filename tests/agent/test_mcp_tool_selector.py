from types import SimpleNamespace
from typing import Any, List, cast

import pytest
from agents.mcp import MCPServer
from agents.mcp.util import MCPUtil

from oai_coding_agent.mcp.mcp_tool_selector import get_filtered_function_tools


class DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture(autouse=True)
def patch_get_function_tools(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """
    By default, stub out MCPUtil.get_function_tools to avoid touching real servers.
    Individual tests will override this.
    """

    async def _fake(server: MCPServer, convert_strict: bool) -> List[Any]:
        return []

    monkeypatch.setattr(MCPUtil, "get_function_tools", _fake)
    return monkeypatch


@pytest.mark.asyncio
async def test_default_mode_no_filter(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # Both edit_file and read_file should pass through in default mode
    async def fake(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [DummyTool("edit_file"), DummyTool("read_file")]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake)
    servers = cast(List[MCPServer], [SimpleNamespace(name="file-system-mcp")])
    tools = await get_filtered_function_tools(servers, mode="default")
    assert {t.name for t in tools} == {"edit_file", "read_file"}


@pytest.mark.asyncio
async def test_plan_mode_filesystem_filter(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # edit_file should be removed in plan mode for file-system-mcp
    async def fake(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [DummyTool("edit_file"), DummyTool("read_file")]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake)
    servers = cast(List[MCPServer], [SimpleNamespace(name="file-system-mcp")])
    tools = await get_filtered_function_tools(servers, mode="plan")
    assert {t.name for t in tools} == {"read_file"}


@pytest.mark.asyncio
async def test_plan_mode_git_filter(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # Only clone_repo and list_branches should remain for mcp-server-git in plan mode
    async def fake(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [
            DummyTool("clone_repo"),
            DummyTool("list_branches"),
            DummyTool("commit"),
        ]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake)
    servers = cast(List[MCPServer], [SimpleNamespace(name="mcp-server-git")])
    tools = await get_filtered_function_tools(servers, mode="plan")
    assert {t.name for t in tools} == {"clone_repo", "list_branches"}


@pytest.mark.asyncio
async def test_github_server_whitelist_default_mode(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # Only the specified whitelist tools (including create/update) should be returned for default mode
    full_allowed = {
        "get_issue",
        "get_issue_comments",
        "create_issue",
        "add_issue_comment",
        "list_issues",
        "update_issue",
        "search_issues",
        "get_pull_request",
        "list_pull_requests",
        "get_pull_request_files",
        "get_pull_request_status",
        "update_pull_request_branch",
        "get_pull_request_comments",
        "get_pull_request_reviews",
        "create_pull_request",
        "add_pull_request_review_comment",
        "update_pull_request",
    }
    names = list(full_allowed) + ["other_tool"]

    async def fake(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [DummyTool(name) for name in names]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake)
    servers = cast(List[MCPServer], [SimpleNamespace(name="github-mcp-server")])

    tools = await get_filtered_function_tools(servers, mode="default")
    assert {t.name for t in tools} == full_allowed


@pytest.mark.asyncio
async def test_github_server_plan_mode_readonly_filter(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # Only read-only tools should be returned for plan mode
    readonly_allowed = {
        "get_issue",
        "get_issue_comments",
        "create_issue",
        "list_issues",
        "search_issues",
        "get_pull_request",
        "list_pull_requests",
        "get_pull_request_files",
        "get_pull_request_status",
        "get_pull_request_comments",
        "get_pull_request_reviews",
    }
    names = list(readonly_allowed) + [
        "add_issue_comment",
        "update_issue",
        "create_pull_request",
        "add_pull_request_review_comment",
        "update_pull_request",
        "update_pull_request_branch",
        "other_tool",
    ]

    async def fake_plan(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [DummyTool(name) for name in names]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake_plan)
    servers = cast(List[MCPServer], [SimpleNamespace(name="github-mcp-server")])

    tools = await get_filtered_function_tools(servers, mode="plan")
    assert {t.name for t in tools} == readonly_allowed


@pytest.mark.asyncio
async def test_atlassian_server_plan_mode_only(
    patch_get_function_tools: pytest.MonkeyPatch,
) -> None:
    # Atlassian MCP server tools should only be available in plan mode
    async def fake(server: MCPServer, convert_strict: bool) -> List[DummyTool]:
        return [
            DummyTool("jira_create_issue"),
            DummyTool("jira_get_issue"),
            DummyTool("confluence_create_page"),
            DummyTool("confluence_get_page"),
        ]

    patch_get_function_tools.setattr(MCPUtil, "get_function_tools", fake)
    servers = cast(List[MCPServer], [SimpleNamespace(name="atlassian-mcp")])

    # Test in plan mode - all tools should be available
    tools = await get_filtered_function_tools(servers, mode="plan")
    assert {t.name for t in tools} == {
        "jira_create_issue",
        "jira_get_issue",
        "confluence_create_page",
        "confluence_get_page",
    }

    # Test in default mode - no tools should be available
    tools = await get_filtered_function_tools(servers, mode="default")
    assert {t.name for t in tools} == set()

    # Test in async mode - no tools should be available
    tools = await get_filtered_function_tools(servers, mode="async")
    assert {t.name for t in tools} == set()
